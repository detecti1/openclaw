import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getMemorySearchManager, type MemoryIndexManager } from "./index.js";

vi.mock("./embeddings.js", () => {
  const embed = (text: string) => {
    const lower = text.toLowerCase();
    const alpha = lower.split("alpha").length - 1;
    const beta = lower.split("beta").length - 1;
    return [alpha, beta];
  };
  return {
    createEmbeddingProvider: async () => ({
      requestedProvider: "openai",
      provider: {
        id: "mock",
        model: "mock-embed",
        embedQuery: async (text: string) => embed(text),
        embedBatch: async (texts: string[]) => texts.map(embed),
      },
    }),
  };
});

describe("memory embedding storage", () => {
  let workspaceDir: string;
  let indexPath: string;
  let manager: MemoryIndexManager | null = null;

  const cfgFor = (workspace: string, dbPath: string) => ({
    agents: {
      defaults: {
        workspace,
        memorySearch: {
          provider: "openai",
          model: "mock-embed",
          store: { path: dbPath, vector: { enabled: false } },
          cache: { enabled: true },
          sync: { watch: false, onSessionStart: false, onSearch: false },
          query: { minScore: 0 },
        },
      },
      list: [{ id: "main", default: true }],
    },
  });

  beforeEach(async () => {
    workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-mem-storage-"));
    indexPath = path.join(workspaceDir, "index.sqlite");
    await fs.mkdir(path.join(workspaceDir, "memory"));
    await fs.writeFile(path.join(workspaceDir, "MEMORY.md"), "alpha beta memory");
  });

  afterEach(async () => {
    if (manager) {
      await manager.close();
      manager = null;
    }
    await fs.rm(workspaceDir, { recursive: true, force: true });
  });

  it("stores chunk and cache embeddings as blobs", async () => {
    const result = await getMemorySearchManager({
      cfg: cfgFor(workspaceDir, indexPath),
      agentId: "main",
    });
    expect(result.manager).not.toBeNull();
    if (!result.manager) {
      throw new Error("manager missing");
    }
    manager = result.manager;

    await manager.sync({ force: true });

    const db = (manager as unknown as { db: { prepare: (sql: string) => unknown } }).db;
    const chunkRow = db.prepare("SELECT typeof(embedding) as t FROM chunks LIMIT 1").get() as
      | { t: string }
      | undefined;
    const cacheRow = db
      .prepare("SELECT typeof(embedding) as t FROM embedding_cache LIMIT 1")
      .get() as { t: string } | undefined;
    const chunkColumn = db
      .prepare("SELECT type FROM pragma_table_info('chunks') WHERE name = 'embedding'")
      .get() as { type: string } | undefined;
    const cacheColumn = db
      .prepare("SELECT type FROM pragma_table_info('embedding_cache') WHERE name = 'embedding'")
      .get() as { type: string } | undefined;

    expect(chunkRow?.t).toBe("blob");
    expect(cacheRow?.t).toBe("blob");
    expect(chunkColumn?.type.toUpperCase()).toBe("BLOB");
    expect(cacheColumn?.type.toUpperCase()).toBe("BLOB");
  });

  it("migrates legacy text(JSON) embeddings to blobs on startup", async () => {
    const result = await getMemorySearchManager({
      cfg: cfgFor(workspaceDir, indexPath),
      agentId: "main",
    });
    expect(result.manager).not.toBeNull();
    if (!result.manager) {
      throw new Error("manager missing");
    }
    manager = result.manager;
    await manager.sync({ force: true });

    const db = (
      manager as unknown as {
        db: {
          prepare: (sql: string) => { run: (...args: unknown[]) => unknown; get: () => unknown };
        };
      }
    ).db;
    db.prepare("UPDATE chunks SET embedding = ?").run("[1,0]");
    db.prepare("UPDATE embedding_cache SET embedding = ?, dims = ?").run("[1,0]", 2);
    const before = db
      .prepare(
        "SELECT (SELECT typeof(embedding) FROM chunks LIMIT 1) AS chunksType, (SELECT typeof(embedding) FROM embedding_cache LIMIT 1) AS cacheType",
      )
      .get() as { chunksType: string; cacheType: string } | undefined;
    expect(before?.chunksType).toBe("text");
    expect(before?.cacheType).toBe("text");

    await manager.close();
    manager = null;

    const reopened = await getMemorySearchManager({
      cfg: cfgFor(workspaceDir, indexPath),
      agentId: "main",
    });
    expect(reopened.manager).not.toBeNull();
    if (!reopened.manager) {
      throw new Error("manager missing");
    }
    manager = reopened.manager;

    const migratedDb = (manager as unknown as { db: { prepare: (sql: string) => unknown } }).db;
    const after = migratedDb
      .prepare(
        "SELECT (SELECT typeof(embedding) FROM chunks LIMIT 1) AS chunksType, (SELECT typeof(embedding) FROM embedding_cache LIMIT 1) AS cacheType",
      )
      .get() as { chunksType: string; cacheType: string } | undefined;

    expect(after?.chunksType).toBe("blob");
    expect(after?.cacheType).toBe("blob");

    const results = await manager.search("alpha");
    expect(results.length).toBeGreaterThan(0);
  });
});
