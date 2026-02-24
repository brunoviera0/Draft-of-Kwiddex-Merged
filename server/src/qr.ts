import { TextEncoder } from "node:util"

const GF_EXP = new Uint8Array(512)
const GF_LOG = new Uint8Array(256)

let x = 1
for (let i = 0; i < 255; i += 1) {
  GF_EXP[i] = x
  GF_LOG[x] = i
  x <<= 1
  if (x & 0x100) {
    x ^= 0x11d
  }
}
for (let i = 255; i < 512; i += 1) {
  GF_EXP[i] = GF_EXP[i - 255]
}

function gfMul(a: number, b: number) {
  if (a === 0 || b === 0) return 0
  return GF_EXP[GF_LOG[a] + GF_LOG[b]]
}

function polyMultiply(a: number[], b: number[]) {
  const result = new Array(a.length + b.length - 1).fill(0)
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < b.length; j += 1) {
      result[i + j] ^= gfMul(a[i], b[j])
    }
  }
  return result
}

function buildGeneratorPoly(ecLength: number) {
  let poly = [1]
  for (let i = 0; i < ecLength; i += 1) {
    poly = polyMultiply(poly, [1, GF_EXP[i]])
  }
  return poly
}

function computeErrorCorrection(data: number[], ecLength: number) {
  const poly = data.concat(new Array(ecLength).fill(0))
  const generator = buildGeneratorPoly(ecLength)
  for (let i = 0; i < data.length; i += 1) {
    const factor = poly[i]
    if (factor === 0) continue
    const logFactor = GF_LOG[factor]
    for (let j = 0; j < generator.length; j += 1) {
      poly[i + j] ^= GF_EXP[(logFactor + GF_LOG[generator[j]]) % 255]
    }
  }
  return poly.slice(poly.length - ecLength)
}

type VersionInfo = {
  version: number
  size: number
  alignmentCenters: number[]
  totalDataCodewords: number
  ecCodewordsPerBlock: number
  group1: { blocks: number; dataCodewords: number }
  group2: { blocks: number; dataCodewords: number }
}

const VERSIONS: VersionInfo[] = [
  // placeholder for 0 index
  {
    version: 0,
    size: 0,
    alignmentCenters: [],
    totalDataCodewords: 0,
    ecCodewordsPerBlock: 0,
    group1: { blocks: 0, dataCodewords: 0 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 1,
    size: 21,
    alignmentCenters: [],
    totalDataCodewords: 16,
    ecCodewordsPerBlock: 10,
    group1: { blocks: 1, dataCodewords: 16 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 2,
    size: 25,
    alignmentCenters: [6, 18],
    totalDataCodewords: 28,
    ecCodewordsPerBlock: 16,
    group1: { blocks: 1, dataCodewords: 28 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 3,
    size: 29,
    alignmentCenters: [6, 22],
    totalDataCodewords: 44,
    ecCodewordsPerBlock: 26,
    group1: { blocks: 1, dataCodewords: 44 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 4,
    size: 33,
    alignmentCenters: [6, 26],
    totalDataCodewords: 64,
    ecCodewordsPerBlock: 18,
    group1: { blocks: 2, dataCodewords: 32 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 5,
    size: 37,
    alignmentCenters: [6, 30],
    totalDataCodewords: 86,
    ecCodewordsPerBlock: 24,
    group1: { blocks: 2, dataCodewords: 43 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 6,
    size: 41,
    alignmentCenters: [6, 34],
    totalDataCodewords: 108,
    ecCodewordsPerBlock: 16,
    group1: { blocks: 4, dataCodewords: 27 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 7,
    size: 45,
    alignmentCenters: [6, 22, 38],
    totalDataCodewords: 124,
    ecCodewordsPerBlock: 18,
    group1: { blocks: 4, dataCodewords: 31 },
    group2: { blocks: 0, dataCodewords: 0 },
  },
  {
    version: 8,
    size: 49,
    alignmentCenters: [6, 24, 42],
    totalDataCodewords: 154,
    ecCodewordsPerBlock: 22,
    group1: { blocks: 2, dataCodewords: 38 },
    group2: { blocks: 2, dataCodewords: 39 },
  },
  {
    version: 9,
    size: 53,
    alignmentCenters: [6, 26, 46],
    totalDataCodewords: 182,
    ecCodewordsPerBlock: 22,
    group1: { blocks: 3, dataCodewords: 36 },
    group2: { blocks: 2, dataCodewords: 37 },
  },
]

function placeFinderPattern(
  modules: (boolean | null)[][],
  reserved: boolean[][],
  row: number,
  col: number,
  size: number
) {
  for (let r = -1; r <= 7; r += 1) {
    for (let c = -1; c <= 7; c += 1) {
      const rr = row + r
      const cc = col + c
      if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue
      if ((r >= 0 && r <= 6 && (c === 0 || c === 6)) || (c >= 0 && c <= 6 && (r === 0 || r === 6))) {
        modules[rr][cc] = true
        reserved[rr][cc] = true
      } else if (r >= 2 && r <= 4 && c >= 2 && c <= 4) {
        modules[rr][cc] = true
        reserved[rr][cc] = true
      } else {
        modules[rr][cc] = false
        reserved[rr][cc] = true
      }
    }
  }
}

function placeTimingPatterns(
  modules: (boolean | null)[][],
  reserved: boolean[][],
  size: number
) {
  for (let i = 0; i < size; i += 1) {
    if (modules[6][i] === null) {
      modules[6][i] = i % 2 === 0
      reserved[6][i] = true
    }
    if (modules[i][6] === null) {
      modules[i][6] = i % 2 === 0
      reserved[i][6] = true
    }
  }
}
function placeAlignmentPattern(
  modules: (boolean | null)[][],
  reserved: boolean[][],
  row: number,
  col: number
) {
  for (let r = -2; r <= 2; r += 1) {
    for (let c = -2; c <= 2; c += 1) {
      const rr = row + r
      const cc = col + c
      if (modules[rr][cc] !== null) continue
      const dist = Math.max(Math.abs(r), Math.abs(c))
      modules[rr][cc] = dist !== 1
      reserved[rr][cc] = true
    }
  }
}

function applyMask(matrix: boolean[][], reserved: boolean[][], maskPattern: number) {
  const size = matrix.length
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      if (reserved[row][col] || matrix[row][col] === null) continue
      if (shouldMask(row, col, maskPattern)) {
        matrix[row][col] = !matrix[row][col]
      }
    }
  }
}

function shouldMask(row: number, col: number, pattern: number) {
  switch (pattern) {
    case 0:
      return (row + col) % 2 === 0
    case 1:
      return row % 2 === 0
    case 2:
      return col % 3 === 0
    case 3:
      return (row + col) % 3 === 0
    default:
      return (Math.floor(row / 2) + Math.floor(col / 3)) % 2 === 0
  }
}

function calcBchFormat(bits: number) {
  let value = bits << 10
  const generator = 0b10100110111
  while (value >= 1 << 10) {
    const shift = Math.floor(Math.log2(value)) - 10
    value ^= generator << shift
  }
  return ((bits << 10) | value) ^ 0b101010000010010
}

function setFormatInfo(
  modules: (boolean | null)[][],
  reserved: boolean[][],
  maskPattern: number
) {
  const formatBits = calcBchFormat((0b01 << 3) | maskPattern)
  const positionsA = [
    [8, 0],
    [8, 1],
    [8, 2],
    [8, 3],
    [8, 4],
    [8, 5],
    [8, 7],
    [8, 8],
    [7, 8],
    [5, 8],
    [4, 8],
    [3, 8],
    [2, 8],
    [1, 8],
    [0, 8],
  ] as const
  const positionsB = [
    [modules.length - 1, 8],
    [modules.length - 2, 8],
    [modules.length - 3, 8],
    [modules.length - 4, 8],
    [modules.length - 5, 8],
    [modules.length - 6, 8],
    [modules.length - 7, 8],
    [8, modules.length - 8],
    [8, modules.length - 7],
    [8, modules.length - 6],
    [8, modules.length - 5],
    [8, modules.length - 4],
    [8, modules.length - 3],
    [8, modules.length - 2],
    [8, modules.length - 1],
  ] as const

  for (let i = 0; i < 15; i += 1) {
    const bit = (formatBits >> (14 - i)) & 1
    const [r1, c1] = positionsA[i]
    const [r2, c2] = positionsB[i]
    modules[r1][c1] = Boolean(bit)
    modules[r2][c2] = Boolean(bit)
    reserved[r1][c1] = true
    reserved[r2][c2] = true
  }
}

function buildDataBytes(value: string, totalDataCodewords: number, version: number) {
  const bytes = new TextEncoder().encode(value)
  const maxBits = totalDataCodewords * 8

  const lengthBits = version <= 9 ? 8 : 16

  const neededBits = 4 + lengthBits + bytes.length * 8
  if (neededBits > maxBits) {
    throw new Error("Value too long for supported QR versions")
  }

  const bits: number[] = []
  const pushBits = (num: number, length: number) => {
    for (let i = length - 1; i >= 0; i -= 1) {
      bits.push((num >> i) & 1)
    }
  }

  pushBits(0b0100, 4) // byte mode
  pushBits(bytes.length, lengthBits)
  for (const byte of bytes) {
    pushBits(byte, 8)
  }

  const remaining = maxBits - bits.length
  if (remaining > 0) {
    const terminator = Math.min(4, remaining)
    for (let i = 0; i < terminator; i += 1) {
      bits.push(0)
    }
  }

  while (bits.length % 8 !== 0) {
    bits.push(0)
  }

  const data: number[] = []
  for (let i = 0; i < bits.length; i += 8) {
    let byte = 0
    for (let j = 0; j < 8; j += 1) {
      byte = (byte << 1) | bits[i + j]
    }
    data.push(byte)
  }

  let padByte = 0xec
  while (data.length < totalDataCodewords) {
    data.push(padByte)
    padByte = padByte === 0xec ? 0x11 : 0xec
  }

  return data
}

function placeDataBits(
  modules: (boolean | null)[][],
  reserved: boolean[][],
  size: number,
  dataCodewords: number[],
  maskPattern: number
) {
  const bits: number[] = []
  for (const byte of dataCodewords) {
    for (let i = 7; i >= 0; i -= 1) {
      bits.push((byte >> i) & 1)
    }
  }

  let bitIndex = 0
  for (let col = size - 1; col > 0; col -= 2) {
    if (col === 6) col -= 1
    for (let rowOffset = 0; rowOffset < size; rowOffset += 1) {
      const row = col % 4 === 1 ? rowOffset : size - 1 - rowOffset
      for (let c = 0; c < 2; c += 1) {
        const targetCol = col - c
        if (reserved[row][targetCol]) continue
        const bit = bitIndex < bits.length ? bits[bitIndex] : 0
        modules[row][targetCol] = Boolean(bit)
        reserved[row][targetCol] = false
        bitIndex += 1
      }
    }
  }

  applyMask(modules as boolean[][], reserved, maskPattern)
}

function calcBchVersion(version: number) {
  let value = version << 12
  const generator = 0x1f25
  while (value >= 1 << 12) {
    const shift = Math.floor(Math.log2(value)) - 12
    value ^= generator << shift
  }
  return (version << 12) | value
}

function setVersionInfo(
  modules: (boolean | null)[][],
  reserved: boolean[][],
  version: number
) {
  if (version < 7) return
  const size = modules.length
  const versionBits = calcBchVersion(version)
  for (let i = 0; i < 18; i += 1) {
    const bit = (versionBits >> (17 - i)) & 1
    const row = Math.floor(i / 3)
    const col = i % 3
    modules[row][size - 11 + col] = Boolean(bit)
    reserved[row][size - 11 + col] = true

    modules[size - 11 + col][row] = Boolean(bit)
    reserved[size - 11 + col][row] = true
  }
}

function interleaveBlocks(
  dataBlocks: number[][],
  ecBlocks: number[][]
) {
  const result: number[] = []
  const maxDataLength = Math.max(...dataBlocks.map((block) => block.length))
  const maxEcLength = Math.max(...ecBlocks.map((block) => block.length))

  for (let i = 0; i < maxDataLength; i += 1) {
    for (const block of dataBlocks) {
      if (i < block.length) {
        result.push(block[i])
      }
    }
  }

  for (let i = 0; i < maxEcLength; i += 1) {
    for (const block of ecBlocks) {
      result.push(block[i])
    }
  }

  return result
}

function selectVersion(value: string) {
  const bytes = new TextEncoder().encode(value)
  for (let version = 1; version < VERSIONS.length; version += 1) {
    const info = VERSIONS[version]
    const maxBits = info.totalDataCodewords * 8
    const lengthBits = version <= 9 ? 8 : 16
    const neededBits = 4 + lengthBits + bytes.length * 8
    if (neededBits <= maxBits) {
      return info
    }
  }
  throw new Error("Value too long for supported QR versions")
}

export function createQrMatrix(value: string) {
  const info = selectVersion(value)
  const size = info.size
  const modules: (boolean | null)[][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => null as boolean | null)
  )
  const reserved: boolean[][] = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => false)
  )

  placeFinderPattern(modules, reserved, 0, 0, size)
  placeFinderPattern(modules, reserved, 0, size - 7, size)
  placeFinderPattern(modules, reserved, size - 7, 0, size)

  placeTimingPatterns(modules, reserved, size)

  for (const row of info.alignmentCenters) {
    for (const col of info.alignmentCenters) {
      if (
        (row <= 7 && col <= 7) ||
        (row <= 7 && col >= size - 8) ||
        (row >= size - 8 && col <= 7)
      ) {
        continue
      }
      placeAlignmentPattern(modules, reserved, row, col)
    }
  }

  modules[size - 8][8] = true
  reserved[size - 8][8] = true

  setVersionInfo(modules, reserved, info.version)

  const dataBytes = buildDataBytes(value, info.totalDataCodewords, info.version)

  const dataBlocks: number[][] = []
  const ecBlocks: number[][] = []
  let offset = 0

  const pushBlocks = (count: number, dataCodewords: number) => {
    for (let i = 0; i < count; i += 1) {
      const block = dataBytes.slice(offset, offset + dataCodewords)
      offset += dataCodewords
      dataBlocks.push(block)
      ecBlocks.push(computeErrorCorrection(block, info.ecCodewordsPerBlock))
    }
  }

  pushBlocks(info.group1.blocks, info.group1.dataCodewords)
  pushBlocks(info.group2.blocks, info.group2.dataCodewords)

  const codewords = interleaveBlocks(dataBlocks, ecBlocks)

  placeDataBits(modules, reserved, size, codewords, 0)
  setFormatInfo(modules, reserved, 0)

  const finalMatrix: boolean[][] = modules.map((row) => row.map((cell) => Boolean(cell)))

  return { size, modules: finalMatrix }
}
