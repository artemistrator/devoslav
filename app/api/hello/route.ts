export async function GET() {
  const body = { message: "Hello from devoslav!" };

  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
    },
  });
}
