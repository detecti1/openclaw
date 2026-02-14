import type { DatabaseSync } from "node:sqlite";
import fs from "node:fs";
import { resolveUserPath } from "../utils.js";
import { requireNodeSqlite } from "./sqlite.js";

export type EmbeddingColumnSchemaState = "blob" | "legacy" | "missing";

export type MemoryEmbeddingSchemaStatus = {
  chunks: EmbeddingColumnSchemaState;
  cache: EmbeddingColumnSchemaState;
  needsLegacyScan: boolean;
};

export function ensureMemoryIndexSchema(params: {
  db: DatabaseSync;
  embeddingCacheTable: string;
  ftsTable: string;
  ftsEnabled: boolean;
}): { ftsAvailable: boolean; ftsError?: string } {
  params.db.exec(`
    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);
  params.db.exec(`
    CREATE TABLE IF NOT EXISTS files (
      path TEXT PRIMARY KEY,
      source TEXT NOT NULL DEFAULT 'memory',
      hash TEXT NOT NULL,
      mtime INTEGER NOT NULL,
      size INTEGER NOT NULL
    );
  `);
  params.db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      path TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'memory',
      start_line INTEGER NOT NULL,
      end_line INTEGER NOT NULL,
      hash TEXT NOT NULL,
      model TEXT NOT NULL,
      text TEXT NOT NULL,
      embedding BLOB NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `);
  params.db.exec(`
    CREATE TABLE IF NOT EXISTS ${params.embeddingCacheTable} (
      provider TEXT NOT NULL,
      model TEXT NOT NULL,
      provider_key TEXT NOT NULL,
      hash TEXT NOT NULL,
      embedding BLOB NOT NULL,
      dims INTEGER,
      updated_at INTEGER NOT NULL,
      PRIMARY KEY (provider, model, provider_key, hash)
    );
  `);
  params.db.exec(
    `CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at ON ${params.embeddingCacheTable}(updated_at);`,
  );

  let ftsAvailable = false;
  let ftsError: string | undefined;
  if (params.ftsEnabled) {
    try {
      params.db.exec(
        `CREATE VIRTUAL TABLE IF NOT EXISTS ${params.ftsTable} USING fts5(\n` +
          `  text,\n` +
          `  id UNINDEXED,\n` +
          `  path UNINDEXED,\n` +
          `  source UNINDEXED,\n` +
          `  model UNINDEXED,\n` +
          `  start_line UNINDEXED,\n` +
          `  end_line UNINDEXED\n` +
          `);`,
      );
      ftsAvailable = true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      ftsAvailable = false;
      ftsError = message;
    }
  }

  ensureColumn(params.db, "files", "source", "TEXT NOT NULL DEFAULT 'memory'");
  ensureColumn(params.db, "chunks", "source", "TEXT NOT NULL DEFAULT 'memory'");
  params.db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);`);
  params.db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);`);

  return { ftsAvailable, ...(ftsError ? { ftsError } : {}) };
}

function ensureColumn(
  db: DatabaseSync,
  table: "files" | "chunks",
  column: string,
  definition: string,
): void {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  if (rows.some((row) => row.name === column)) {
    return;
  }
  db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
}

export function getMemoryEmbeddingSchemaStatus(params: {
  db: DatabaseSync;
  embeddingCacheTable: string;
}): MemoryEmbeddingSchemaStatus {
  const chunks = getEmbeddingColumnSchemaState(params.db, "chunks");
  const cache = getEmbeddingColumnSchemaState(params.db, params.embeddingCacheTable);
  const needsLegacyScan = chunks !== "blob" || cache !== "blob";
  return { chunks, cache, needsLegacyScan };
}

export function readMemoryEmbeddingSchemaStatusFromPath(params: {
  dbPath: string;
  embeddingCacheTable: string;
}): MemoryEmbeddingSchemaStatus | null {
  const resolvedPath = resolveUserPath(params.dbPath);
  if (!fs.existsSync(resolvedPath)) {
    return null;
  }
  try {
    const { DatabaseSync } = requireNodeSqlite();
    const db = new DatabaseSync(resolvedPath, { allowExtension: false });
    try {
      return getMemoryEmbeddingSchemaStatus({
        db,
        embeddingCacheTable: params.embeddingCacheTable,
      });
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

function getEmbeddingColumnSchemaState(
  db: DatabaseSync,
  table: string,
): EmbeddingColumnSchemaState {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
    name: string;
    type?: string;
  }>;
  const embeddingColumn = columns.find((entry) => entry.name === "embedding");
  if (!embeddingColumn) {
    return "missing";
  }
  const declaredType = embeddingColumn.type?.trim().toUpperCase() ?? "";
  return declaredType.includes("BLOB") ? "blob" : "legacy";
}
