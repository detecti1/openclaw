import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getMemorySearchManager, type MemoryIndexManager } from "./index.js";
import { migrateMemoryStoreToBlob } from "./store-migrate.js";

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

describe("migrateMemoryStoreToBlob", () => {
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
    workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-mem-migrate-"));
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

  it("backs up and converts text embeddings to blobs", async () => {
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

    await manager.close();
    manager = null;

    const migrated = await migrateMemoryStoreToBlob({
      dbPath: indexPath,
      keepBackup: true,
      vector: { enabled: false },
    });
    expect(migrated.backupPath).toBeTruthy();
    if (migrated.backupPath) {
      const backupStat = await fs.stat(migrated.backupPath);
      expect(backupStat.isFile()).toBe(true);
    }
    expect(migrated.chunksConverted).toBeGreaterThan(0);
    expect(migrated.cacheConverted).toBeGreaterThan(0);

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
    const row = migratedDb
      .prepare(
        "SELECT (SELECT typeof(embedding) FROM chunks LIMIT 1) AS chunksType, (SELECT typeof(embedding) FROM embedding_cache LIMIT 1) AS cacheType",
      )
      .get() as { chunksType: string; cacheType: string } | undefined;
    expect(row?.chunksType).toBe("blob");
    expect(row?.cacheType).toBe("blob");
  });

  it("fails when database file is missing", async () => {
    await expect(
      migrateMemoryStoreToBlob({
        dbPath: path.join(workspaceDir, "missing.sqlite"),
        vector: { enabled: false },
      }),
    ).rejects.toThrow(/database not found/i);
  });
});
