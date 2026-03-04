import mysql, { Pool, PoolOptions } from 'mysql2/promise';

const required = ['DB_USER', 'DB_NAME'] as const;

let pool: Pool | null = null;
let usersTableReady = false;

const assertRequiredConfig = () => {
  for (const key of required) {
    if (!process.env[key]?.trim()) {
      throw new Error(`${key} is not configured.`);
    }
  }
};

const createPoolConfig = (): PoolOptions => {
  assertRequiredConfig();

  const instanceConnectionName = process.env.INSTANCE_CONNECTION_NAME?.trim();
  const user = process.env.DB_USER!.trim();
  const database = process.env.DB_NAME!.trim();

  const commonConfig: PoolOptions = {
    user,
    database,
    password: process.env.DB_PASS ?? process.env.DB_PASSWORD ?? '',
    waitForConnections: true,
    connectionLimit: Number(process.env.DB_CONNECTION_LIMIT ?? 10),
    queueLimit: 0,
    enableKeepAlive: true,
  };

  if (instanceConnectionName) {
    return {
      ...commonConfig,
      socketPath: `/cloudsql/${instanceConnectionName}`,
    };
  }

  const host = process.env.DB_HOST?.trim();
  if (host) {
    return {
      ...commonConfig,
      host,
      port: Number(process.env.DB_PORT ?? 3306),
    };
  }

  throw new Error(
    'Database connection is not configured. Set INSTANCE_CONNECTION_NAME for Cloud SQL Unix socket, or set DB_HOST (and optional DB_PORT) for local TCP.',
  );
};

export const getDbPool = (): Pool => {
  if (!pool) {
    pool = mysql.createPool(createPoolConfig());
  }

  return pool;
};

export const ensureUsersTable = async () => {
  if (usersTableReady) {
    return;
  }

  await getDbPool().execute(`
    CREATE TABLE IF NOT EXISTS users (
      id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
      email VARCHAR(255) UNIQUE NOT NULL,
      password_hash VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
  `);

  usersTableReady = true;
};

export const checkDbHealth = async () => {
  const [rows] = await getDbPool().query('SELECT 1 AS ok;');
  return rows;
};

export const closeDbPool = async () => {
  if (!pool) return;
  await pool.end();
  pool = null;
  usersTableReady = false;
};
