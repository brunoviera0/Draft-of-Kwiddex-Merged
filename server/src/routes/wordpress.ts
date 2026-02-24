/**
 * routes/wordpress.ts: Headless WordPress CMS proxy.
 * Fetches posts/pages from WordPress REST API, returns sanitized JSON.
 * Set WORDPRESS_URL to the WP site once established (for example, https://blog.kwiddex.com).
 */


import { Router, type Request, type Response } from "express"

const wpRouter = Router()
const WORDPRESS_URL = (process.env.WORDPRESS_URL || "").replace(/\/+$/, "")
const WP_API = `${WORDPRESS_URL}/wp-json/wp/v2`
const CACHE_TTL_MS = 5 * 60 * 1000

type CacheEntry = { data: unknown; cachedAt: number }
const cache = new Map<string, CacheEntry>()

function getCached(key: string): unknown | null {
  const entry = cache.get(key)
  if (!entry) return null
  if (Date.now() - entry.cachedAt > CACHE_TTL_MS) { cache.delete(key); return null }
  return entry.data
}

async function wpFetch(fullUrl: string): Promise<unknown> {
  if (!WORDPRESS_URL) throw new Error("WORDPRESS_URL is not configured.")
  const cached = getCached(fullUrl)
  if (cached) return cached
  const response = await fetch(fullUrl, { headers: { Accept: "application/json" } })
  if (!response.ok) throw new Error(`WordPress API returned ${response.status}`)
  const data = await response.json()
  cache.set(fullUrl, { data, cachedAt: Date.now() })
  return data
}

function sanitizePost(post: any) {
  return {
    id: post.id,
    slug: post.slug,
    title: post.title?.rendered || "",
    excerpt: post.excerpt?.rendered || "",
    content: post.content?.rendered || "",
    date: post.date,
    modified: post.modified,
    featuredImage: post._embedded?.["wp:featuredmedia"]?.[0]?.source_url || null,
    author: post._embedded?.author?.[0]?.name || null,
    categories: (post._embedded?.["wp:term"]?.[0] || []).map((c: any) => ({
      id: c.id, name: c.name, slug: c.slug,
    })),
  }
}

wpRouter.get("/posts", async (req: Request, res: Response) => {
  if (!WORDPRESS_URL) return res.status(503).json({ error: "Blog is not configured." })
  try {
    const url = new URL(`${WP_API}/posts`)
    url.searchParams.set("_embed", "true")
    url.searchParams.set("page", String(req.query.page || 1))
    url.searchParams.set("per_page", String(req.query.per_page || 10))
    const posts = (await wpFetch(url.toString())) as any[]
    return res.json(posts.map(sanitizePost))
  } catch (error: any) {
    console.error("[wp] Posts failed:", error?.message)
    return res.status(502).json({ error: "Unable to fetch blog posts." })
  }
})

wpRouter.get("/posts/:slug", async (req: Request, res: Response) => {
  if (!WORDPRESS_URL) return res.status(503).json({ error: "Blog is not configured." })
  try {
    const url = new URL(`${WP_API}/posts`)
    url.searchParams.set("slug", req.params.slug)
    url.searchParams.set("_embed", "true")
    const posts = (await wpFetch(url.toString())) as any[]
    if (!posts.length) return res.status(404).json({ error: "Post not found." })
    return res.json(sanitizePost(posts[0]))
  } catch (error: any) {
    console.error("[wp] Post failed:", error?.message)
    return res.status(502).json({ error: "Unable to fetch blog post." })
  }
})

wpRouter.get("/pages/:slug", async (req: Request, res: Response) => {
  if (!WORDPRESS_URL) return res.status(503).json({ error: "Blog is not configured." })
  try {
    const url = new URL(`${WP_API}/pages`)
    url.searchParams.set("slug", req.params.slug)
    url.searchParams.set("_embed", "true")
    const pages = (await wpFetch(url.toString())) as any[]
    if (!pages.length) return res.status(404).json({ error: "Page not found." })
    return res.json(sanitizePost(pages[0]))
  } catch (error: any) {
    console.error("[wp] Page failed:", error?.message)
    return res.status(502).json({ error: "Unable to fetch page." })
  }
})

wpRouter.get("/categories", async (_req: Request, res: Response) => {
  if (!WORDPRESS_URL) return res.status(503).json({ error: "Blog is not configured." })
  try {
    const url = new URL(`${WP_API}/categories`)
    url.searchParams.set("per_page", "100")
    url.searchParams.set("hide_empty", "true")
    const cats = (await wpFetch(url.toString())) as any[]
    return res.json(cats.map((c: any) => ({ id: c.id, name: c.name, slug: c.slug, count: c.count })))
  } catch (error: any) {
    console.error("[wp] Categories failed:", error?.message)
    return res.status(502).json({ error: "Unable to fetch categories." })
  }
})

export default wpRouter
