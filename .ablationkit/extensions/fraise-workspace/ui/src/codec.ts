/** Undo the Python side's packing: gzip+base64 of JSON in which uniform object lists are columnar. */

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

function unpack(value: Json): Json {
  if (Array.isArray(value)) return value.map(unpack);
  if (value && typeof value === "object") {
    const cols = value["__cols__"];
    const rows = value["__rows__"];
    if (Array.isArray(cols) && Array.isArray(rows)) {
      return rows.map((row) =>
        Object.fromEntries(cols.map((c, i) => [String(c), unpack((row as Json[])[i] ?? null)])),
      );
    }
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, unpack(v)]));
  }
  return value;
}

export async function decode(data: { encoding?: string; payload?: string } | undefined): Promise<unknown> {
  if (data?.encoding !== "gzip+base64:packed-json" || typeof data.payload !== "string")
    throw new Error("unexpected result encoding");
  const bytes = Uint8Array.from(atob(data.payload), (c) => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  const text = await new Response(stream).text();
  return unpack(JSON.parse(text) as Json);
}
