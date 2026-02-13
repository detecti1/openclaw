import type { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveUserPath } from "../utils.js";
import { embeddingToBlob, parseEmbedding } from "./internal.js";
import { ensureMemoryIndexSchema } from "./memory-schema.js";
import { loadSqliteVecExtension } from "./sqlite-vec.js";
import { requireNodeSqlite } from "./sqlite.js";

const EMBEDDING_CACHE_TABLE = "embedding_cache";
const FTS_TABLE = "chunks_fts";
const VECTOR_TABLE = "chunks_vec";
const META_KEY = "memory_index_meta_v1";
const COPY_BATCH_SIZE = 500;

export type MemoryStoreMigrateOptions = {
  dbPath: string;
  keepBackup?: boolean;
  vector?: {
    enabled: boolean;
    extensionPath?: string;
  };
};

export type MemoryStoreMigrateResult = {
  dbPath: string;
  backupPath?: string;
  chunksConverted: number;
  cacheConverted: number;
  vectorRebuilt: boolean;
  ftsRebuilt: boolean;
};

export async function migrateMemoryStoreToBlob(
  opts: MemoryStoreMigrateOptions,
): Promise<MemoryStoreMigrateResult> {
  const dbPath = resolveUserPath(opts.dbPath);
  await assertDbExists(dbPath);

  const tempPath = `${dbPath}.migrate-${randomUUID()}.tmp`;
  const backupPath = `${dbPath}.backup-${randomUUID()}`;
  const { DatabaseSync } = requireNodeSqlite();
  const sourceDb = new DatabaseSync(dbPath, { allowExtension: true });

  let targetDb: DatabaseSync | null = null;
  let sourceTxOpen = false;
  let targetTxOpen = false;
  let chunksConverted = 0;
  let cacheConverted = 0;
  let vectorRebuilt = false;
  let rebuiltVectorDims: number | null = null;
  let ftsRebuilt = false;

  try {
    sourceDb.exec("PRAGMA busy_timeout = 1000");
    try {
      sourceDb.exec("BEGIN EXCLUSIVE");
      sourceTxOpen = true;
    } catch (err) {
      throw new Error(
        `Memory database is busy (another process is using it). Stop the gateway first, then retry. ${formatCause(
          err,
        )}`,
        { cause: err },
      );
    }

    targetDb = new DatabaseSync(tempPath, { allowExtension: true });
    const sourceHasFts = tableExists(sourceDb, FTS_TABLE);
    const sourceMetaVectorDims = readMetaVectorDims(sourceDb);
    const sourceVectorTableDims = readVectorTableDims(sourceDb);
    const vectorRequired =
      (sourceMetaVectorDims !== null && sourceMetaVectorDims > 0) ||
      (sourceVectorTableDims !== null && sourceVectorTableDims > 0);

    const schemaResult = ensureMemoryIndexSchema({
      db: targetDb,
      embeddingCacheTable: EMBEDDING_CACHE_TABLE,
      ftsTable: FTS_TABLE,
      ftsEnabled: sourceHasFts,
    });

    copyMetaTable(sourceDb, targetDb);
    copyFilesTable(sourceDb, targetDb);

    targetDb.exec("BEGIN");
    targetTxOpen = true;

    const copiedChunks = copyChunksTable(sourceDb, targetDb);
    chunksConverted = copiedChunks.converted;
    const copiedCache = copyEmbeddingCacheTable(sourceDb, targetDb);
    cacheConverted = copiedCache.converted;

    const vectorEnabled = opts.vector?.enabled === true;
    if (vectorEnabled) {
      const loaded = await loadSqliteVecExtension({
        db: targetDb,
        extensionPath: opts.vector?.extensionPath,
      });
      if (!loaded.ok) {
        if (vectorRequired) {
          throw new Error(
            `Vector index exists but sqlite-vec could not be loaded; refusing to continue to avoid forcing reindex. ${loaded.error ?? ""}`.trim(),
          );
        }
      } else {
        const vectorDims =
          copiedChunks.vectorDims ?? sourceMetaVectorDims ?? sourceVectorTableDims ?? null;
        if (vectorDims && vectorDims > 0) {
          rebuildVectorTable(targetDb, vectorDims);
          vectorRebuilt = true;
          rebuiltVectorDims = vectorDims;
        }
      }
    }

    if (schemaResult.ftsAvailable) {
      rebuildFtsTable(targetDb);
      ftsRebuilt = true;
    }

    setMetaVectorDims(targetDb, vectorRebuilt ? rebuiltVectorDims : null);

    targetDb.exec("COMMIT");
    targetTxOpen = false;
    targetDb.close();
    targetDb = null;

    sourceDb.exec("COMMIT");
    sourceTxOpen = false;
    sourceDb.close();

    await swapWithBackup(dbPath, tempPath, backupPath);
    if (opts.keepBackup === false) {
      await removeIndexFiles(backupPath);
      return {
        dbPath,
        chunksConverted,
        cacheConverted,
        vectorRebuilt,
        ftsRebuilt,
      };
    }
    return {
      dbPath,
      backupPath,
      chunksConverted,
      cacheConverted,
      vectorRebuilt,
      ftsRebuilt,
    };
  } catch (err) {
    if (targetDb) {
      try {
        if (targetTxOpen) {
          targetDb.exec("ROLLBACK");
        }
      } catch {}
      try {
        targetDb.close();
      } catch {}
    }
    try {
      if (sourceTxOpen) {
        sourceDb.exec("ROLLBACK");
      }
    } catch {}
    try {
      sourceDb.close();
    } catch {}
    await removeIndexFiles(tempPath);
    throw err;
  }
}

async function assertDbExists(dbPath: string): Promise<void> {
  try {
    const stat = await fs.stat(dbPath);
    if (!stat.isFile()) {
      throw new Error(`not a file: ${dbPath}`);
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      throw new Error(`Memory database not found at ${dbPath}`, { cause: err });
    }
    throw err;
  }
}

function copyMetaTable(sourceDb: DatabaseSync, targetDb: DatabaseSync): void {
  if (!tableExists(sourceDb, "meta")) {
    return;
  }
  const rows = sourceDb.prepare("SELECT key, value FROM meta").all() as Array<{
    key: string;
    value: string;
  }>;
  const insert = targetDb.prepare("INSERT INTO meta (key, value) VALUES (?, ?)");
  for (const row of rows) {
    insert.run(row.key, row.value);
  }
}

function copyFilesTable(sourceDb: DatabaseSync, targetDb: DatabaseSync): void {
  if (!tableExists(sourceDb, "files")) {
    return;
  }
  const columns = getTableColumns(sourceDb, "files");
  const sourceExpr = columns.has("source") ? "source" : "'memory' AS source";
  const rows = sourceDb
    .prepare(`SELECT path, ${sourceExpr}, hash, mtime, size FROM files`)
    .all() as Array<{
    path: string;
    source: string;
    hash: string;
    mtime: number;
    size: number;
  }>;
  const insert = targetDb.prepare(
    "INSERT INTO files (path, source, hash, mtime, size) VALUES (?, ?, ?, ?, ?)",
  );
  for (const row of rows) {
    insert.run(row.path, row.source, row.hash, row.mtime, row.size);
  }
}

function copyChunksTable(
  sourceDb: DatabaseSync,
  targetDb: DatabaseSync,
): { converted: number; vectorDims: number | null } {
  if (!tableExists(sourceDb, "chunks")) {
    return { converted: 0, vectorDims: null };
  }
  const columns = getTableColumns(sourceDb, "chunks");
  const sourceExpr = columns.has("source") ? "source" : "'memory' AS source";
  const select = sourceDb.prepare(
    `SELECT rowid, id, path, ${sourceExpr}, start_line, end_line, hash, model, text, embedding, typeof(embedding) AS embedding_type, updated_at
       FROM chunks
      WHERE rowid > ?
      ORDER BY rowid ASC
      LIMIT ?`,
  );
  const insert = targetDb.prepare(
    "INSERT INTO chunks (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  );

  let converted = 0;
  let lastRowid = 0;
  let vectorDims: number | null = null;
  while (true) {
    const rows = select.all(lastRowid, COPY_BATCH_SIZE) as Array<{
      rowid: number;
      id: string;
      path: string;
      source: string;
      start_line: number;
      end_line: number;
      hash: string;
      model: string;
      text: string;
      embedding: string | Uint8Array | ArrayBuffer;
      embedding_type: string;
      updated_at: number;
    }>;
    if (rows.length === 0) {
      break;
    }
    for (const row of rows) {
      const blob = toEmbeddingBlob(row.embedding, row.embedding_type);
      if (row.embedding_type === "text") {
        converted += 1;
      }
      if (vectorDims === null && blob.length > 0 && blob.length % 4 === 0) {
        vectorDims = blob.length / 4;
      }
      insert.run(
        row.id,
        row.path,
        row.source,
        row.start_line,
        row.end_line,
        row.hash,
        row.model,
        row.text,
        blob,
        row.updated_at,
      );
      lastRowid = row.rowid;
    }
  }

  return { converted, vectorDims };
}

function copyEmbeddingCacheTable(
  sourceDb: DatabaseSync,
  targetDb: DatabaseSync,
): { converted: number } {
  if (!tableExists(sourceDb, EMBEDDING_CACHE_TABLE)) {
    return { converted: 0 };
  }
  const columns = getTableColumns(sourceDb, EMBEDDING_CACHE_TABLE);
  const dimsExpr = columns.has("dims") ? "dims" : "NULL AS dims";
  const select = sourceDb.prepare(
    `SELECT rowid, provider, model, provider_key, hash, embedding, typeof(embedding) AS embedding_type, ${dimsExpr}, updated_at
       FROM ${EMBEDDING_CACHE_TABLE}
      WHERE rowid > ?
      ORDER BY rowid ASC
      LIMIT ?`,
  );
  const insert = targetDb.prepare(
    `INSERT INTO ${EMBEDDING_CACHE_TABLE} (provider, model, provider_key, hash, embedding, dims, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  );
  let converted = 0;
  let lastRowid = 0;
  while (true) {
    const rows = select.all(lastRowid, COPY_BATCH_SIZE) as Array<{
      rowid: number;
      provider: string;
      model: string;
      provider_key: string;
      hash: string;
      embedding: string | Uint8Array | ArrayBuffer;
      embedding_type: string;
      dims: number | null;
      updated_at: number;
    }>;
    if (rows.length === 0) {
      break;
    }
    for (const row of rows) {
      const blob = toEmbeddingBlob(row.embedding, row.embedding_type);
      if (row.embedding_type === "text") {
        converted += 1;
      }
      insert.run(
        row.provider,
        row.model,
        row.provider_key,
        row.hash,
        blob,
        row.dims,
        row.updated_at,
      );
      lastRowid = row.rowid;
    }
  }
  return { converted };
}

function rebuildVectorTable(targetDb: DatabaseSync, dims: number): void {
  targetDb.exec(`DROP TABLE IF EXISTS ${VECTOR_TABLE}`);
  targetDb.exec(
    `CREATE VIRTUAL TABLE IF NOT EXISTS ${VECTOR_TABLE} USING vec0(\n` +
      `  id TEXT PRIMARY KEY,\n` +
      `  embedding FLOAT[${dims}]\n` +
      `)`,
  );
  const insert = targetDb.prepare(`INSERT INTO ${VECTOR_TABLE} (id, embedding) VALUES (?, ?)`);
  const rows = targetDb
    .prepare("SELECT id, embedding FROM chunks WHERE length(embedding) = ?")
    .all(dims * 4) as Array<{ id: string; embedding: Uint8Array }>;
  for (const row of rows) {
    insert.run(row.id, row.embedding);
  }
}

function rebuildFtsTable(targetDb: DatabaseSync): void {
  targetDb.exec(`DELETE FROM ${FTS_TABLE}`);
  targetDb.exec(
    `INSERT INTO ${FTS_TABLE} (text, id, path, source, model, start_line, end_line)\n` +
      `SELECT text, id, path, source, model, start_line, end_line FROM chunks`,
  );
}

function setMetaVectorDims(targetDb: DatabaseSync, dims: number | null): void {
  const row = targetDb.prepare("SELECT value FROM meta WHERE key = ?").get(META_KEY) as
    | { value?: string }
    | undefined;
  if (!row?.value) {
    return;
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(row.value) as Record<string, unknown>;
  } catch {
    return;
  }
  if (dims && dims > 0) {
    parsed.vectorDims = dims;
  } else {
    delete parsed.vectorDims;
  }
  targetDb
    .prepare(
      "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
    )
    .run(META_KEY, JSON.stringify(parsed));
}

function readMetaVectorDims(db: DatabaseSync): number | null {
  if (!tableExists(db, "meta")) {
    return null;
  }
  const row = db.prepare("SELECT value FROM meta WHERE key = ?").get(META_KEY) as
    | { value?: string }
    | undefined;
  if (!row?.value) {
    return null;
  }
  try {
    const parsed = JSON.parse(row.value) as { vectorDims?: unknown };
    const dims = parsed.vectorDims;
    if (typeof dims === "number" && Number.isFinite(dims) && dims > 0) {
      return Math.floor(dims);
    }
  } catch {}
  return null;
}

function readVectorTableDims(db: DatabaseSync): number | null {
  if (!tableExists(db, VECTOR_TABLE)) {
    return null;
  }
  const row = db
    .prepare("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(VECTOR_TABLE) as { sql?: string } | undefined;
  const sql = row?.sql ?? "";
  const match = /FLOAT\[(\d+)\]/i.exec(sql);
  if (!match) {
    return null;
  }
  const dims = Number(match[1]);
  if (!Number.isFinite(dims) || dims <= 0) {
    return null;
  }
  return Math.floor(dims);
}

function tableExists(db: DatabaseSync, tableName: string): boolean {
  const row = db
    .prepare("SELECT 1 AS ok FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(tableName) as { ok?: number } | undefined;
  return row?.ok === 1;
}

function getTableColumns(db: DatabaseSync, tableName: string): Set<string> {
  const rows = db.prepare(`PRAGMA table_info(${tableName})`).all() as Array<{ name: string }>;
  return new Set(rows.map((row) => row.name));
}

function toEmbeddingBlob(
  raw: string | Uint8Array | ArrayBuffer | null | undefined,
  type: string,
): Uint8Array {
  if (type === "blob") {
    if (raw instanceof Uint8Array) {
      return raw;
    }
    if (raw instanceof ArrayBuffer) {
      return new Uint8Array(raw);
    }
  }
  return embeddingToBlob(parseEmbedding(raw));
}

async function swapWithBackup(
  targetPath: string,
  tempPath: string,
  backupPath: string,
): Promise<void> {
  await moveIndexFiles(targetPath, backupPath);
  try {
    await moveIndexFiles(tempPath, targetPath);
  } catch (err) {
    await moveIndexFiles(backupPath, targetPath);
    throw err;
  }
}

async function moveIndexFiles(sourceBase: string, targetBase: string): Promise<void> {
  const suffixes = ["", "-wal", "-shm"];
  for (const suffix of suffixes) {
    const source = `${sourceBase}${suffix}`;
    const target = `${targetBase}${suffix}`;
    try {
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.rename(source, target);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
        throw err;
      }
    }
  }
}

async function removeIndexFiles(basePath: string): Promise<void> {
  const suffixes = ["", "-wal", "-shm"];
  await Promise.all(suffixes.map((suffix) => fs.rm(`${basePath}${suffix}`, { force: true })));
}

function formatCause(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
