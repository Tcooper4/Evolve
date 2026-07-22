/**
 * Stable front door for Evolve mobile (quick Cloudflare tunnel).
 * GET /           → redirect to current trycloudflare URL (or "start PC" page)
 * GET /api/url    → JSON { url, updated_at }
 * POST /update    → set URL (header X-Evolve-Secret, body { "url": "https://..." })
 */
export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;

    if (path === "/update" && request.method === "POST") {
      const secret = request.headers.get("X-Evolve-Secret") || "";
      if (!env.UPDATE_SECRET || secret !== env.UPDATE_SECRET) {
        return new Response("Unauthorized", { status: 401 });
      }
      let body;
      try {
        body = await request.json();
      } catch {
        return new Response("Bad JSON", { status: 400 });
      }
      const tunnel = String(body.url || "").trim().replace(/\/+$/, "");
      if (!/^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/i.test(tunnel)) {
        return new Response("Invalid tunnel URL", { status: 400 });
      }
      await env.TUNNEL.put("current", tunnel);
      await env.TUNNEL.put("updated_at", new Date().toISOString());
      return Response.json({ ok: true, url: tunnel });
    }

    if (path === "/api/url") {
      const current = await env.TUNNEL.get("current");
      const updatedAt = await env.TUNNEL.get("updated_at");
      return Response.json({ url: current, updated_at: updatedAt });
    }

    const current = await env.TUNNEL.get("current");
    if (!current) {
      return html(
        503,
        `<h1>Evolve</h1>
         <p>Your home PC tunnel is not connected.</p>
         <p>On the PC, run <code>scripts\\start-evolve-mobile.ps1</code> (Docker + cloudflared).</p>`,
      );
    }

    if (path === "/" || path === "/go") {
      return Response.redirect(current, 302);
    }

    return new Response("Not found", { status: 404 });
  },
};

function html(status, inner) {
  return new Response(
    `<!DOCTYPE html><html lang="en"><head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"/>
      <meta name="theme-color" content="#05070d"/>
      <title>Evolve</title>
      <style>
        body{font-family:system-ui,sans-serif;background:#05070d;color:#eef2f8;
          padding:1.5rem;line-height:1.5;max-width:28rem;margin:0 auto}
        code{background:#151d2e;padding:.15rem .4rem;border-radius:4px;font-size:.9em}
      </style></head><body>${inner}</body></html>`,
    { status, headers: { "Content-Type": "text/html; charset=utf-8" } },
  );
}
