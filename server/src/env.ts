import fs from "node:fs"
import path from "node:path"

const moduleDir = __dirname
const serverRoot = path.resolve(moduleDir, "..")
const projectRoot = path.resolve(serverRoot, "..")
const envFiles = [
  path.join(serverRoot, ".env.local"),
  path.join(serverRoot, ".env"),
  path.join(projectRoot, ".env.local"),
  path.join(projectRoot, ".env"),
]

for (const file of envFiles) {
  if (!file) continue
  tryLoadEnvFile(file)
}

if (process.env.FASTAPI_URL === undefined) {
  process.env.FASTAPI_URL = "http://localhost:8000"
}

if (process.env.USE_MONTE_CARLO === undefined) {
  process.env.USE_MONTE_CARLO = "true"
}

if (process.env.MC_SAMPLES === undefined) {
  process.env.MC_SAMPLES = "30"
}

if (process.env.MAX_UPLOAD_MB === undefined) {
  process.env.MAX_UPLOAD_MB = "15"
}

function tryLoadEnvFile(filePath: string) {
  if (!fs.existsSync(filePath)) return
  try {
    const contents = fs.readFileSync(filePath, "utf8")
    parseEnv(contents)
  } catch (error) {
    console.warn(`Failed to read env file ${filePath}:`, error)
  }
}

function parseEnv(source: string) {
  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue

    const equalsIndex = line.indexOf("=")
    if (equalsIndex === -1) continue

    const key = line.slice(0, equalsIndex).trim()
    if (!key) continue
    if (Object.prototype.hasOwnProperty.call(process.env, key)) continue

    let value = line.slice(equalsIndex + 1).trim()
    if (value.startsWith('"') && value.endsWith('"')) {
      value = value.slice(1, -1).replace(/\\"/g, '"')
    } else if (value.startsWith("'") && value.endsWith("'")) {
      value = value.slice(1, -1).replace(/\\'/g, "'")
    }
    if (value === "null") value = ""

    process.env[key] = value
  }
}
