# Moving the vector store to SAP HANA Cloud - honest status

`HanaVectorStore` (`rag/vectorstore.py`) targets the HANA Cloud vector engine through plain SQL:

```sql
CREATE TABLE "POLICY_CHUNKS" (ID NVARCHAR(200) PRIMARY KEY, VEC REAL_VECTOR(<dim>));
INSERT INTO "POLICY_CHUNKS" (ID, VEC) VALUES (?, TO_REAL_VECTOR(?));          -- vector bound as a JSON array string
SELECT TOP k ID, COSINE_SIMILARITY(VEC, TO_REAL_VECTOR(?)) AS SCORE
  FROM "POLICY_CHUNKS" ORDER BY SCORE DESC;
```

**Not verified against a real instance.** The unit tests (`tests/test_vectorstore.py`) only check the generated statements, that vectors are passed as bound parameters (never interpolated), and that unsafe table names are rejected. Syntax and function names come from general knowledge of the vector engine - confirm them against the current SAP HANA Cloud documentation before relying on them.

## Suggested path
1. Start a HANA Cloud instance (a free-tier instance is available in a BTP trial account; it is stopped nightly and must be restarted daily - check current terms).
2. `pip install hdbcli`, open a connection, construct `HanaVectorStore(conn, "POLICY_CHUNKS", dim)`.
3. Replace `LsaEmbedder` with a real embedding model so `dim` matches the model, then re-run `python -m evals.run_evals` and compare against the table in `docs/eval_report.md`.
4. Only then describe it as "deployed on HANA Cloud" - and cite the measured numbers from that run, not these.
