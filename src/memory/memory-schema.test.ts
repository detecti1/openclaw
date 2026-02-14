import { afterEach, describe, expect, it } from "vitest";
import {
  ensureMemoryIndexSchema,
  getMemoryEmbeddingSchemaStatus,
  type MemoryEmbeddingSchemaStatus,
} from "./memory-schema.js";
import { requireNodeSqlite } from "./sqlite.js";

describe("memory embedding schema status", () => {
  const openDbs: Array<import("node:sqlite").DatabaseSync> = [];

  afterEach(() => {
    for (const db of openDbs.splice(0, openDbs.length)) {
      db.close();
    }
  });

  const openDb = () => {
    const { DatabaseSync } = requireNodeSqlite();
    const db = new DatabaseSync(":memory:");
    openDbs.push(db);
    return db;
  };

  const readStatus = (db: import("node:sqlite").DatabaseSync): MemoryEmbeddingSchemaStatus =>
    getMemoryEmbeddingSchemaStatus({ db, embeddingCacheTable: "embedding_cache" });

  it("reports blob schema for new stores", () => {
    const db = openDb();
    ensureMemoryIndexSchema({
      db,
      embeddingCacheTable: "embedding_cache",
      ftsTable: "chunks_fts",
      ftsEnabled: false,
    });

    expect(readStatus(db)).toEqual({
      chunks: "blob",
      cache: "blob",
      needsLegacyScan: false,
    });
  });

  it("reports legacy schema when embedding columns are not declared as blob", () => {
    const db = openDb();
    db.exec(`
      CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        embedding TEXT NOT NULL
      );
    `);
    db.exec(`
      CREATE TABLE embedding_cache (
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        provider_key TEXT NOT NULL,
        hash TEXT NOT NULL,
        embedding TEXT NOT NULL,
        PRIMARY KEY (provider, model, provider_key, hash)
      );
    `);

    expect(readStatus(db)).toEqual({
      chunks: "legacy",
      cache: "legacy",
      needsLegacyScan: true,
    });
  });
});
