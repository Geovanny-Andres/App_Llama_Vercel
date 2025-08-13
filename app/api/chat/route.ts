// app/api/chat/route.ts
// ===============================================================
// Endpoint de chat (App Router, Next.js) con LlamaIndex + OpenAI.
// - Recibe el historial desde el cliente.
// - Normaliza roles/contenido.
// - Inserta un system prompt para ser fiel al PDF.
// - Llama al motor de chat con firma POSICIONAL (message, history, stream).
// - Devuelve la respuesta en streaming.
// ===============================================================

import { Message, StreamingTextResponse } from "ai";
import { OpenAI } from "llamaindex";
import { NextRequest, NextResponse } from "next/server";
import { createChatEngine } from "./engine";
import { LlamaIndexStream } from "./llamaindex-stream";

// Tipado local que sí acepta LlamaIndex (evitamos roles no soportados)
type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
};

// Forzamos runtime Node y respuestas dinámicas (sin caché)
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    // 1) Body -> messages (historial completo, último = usuario)
    const body = await request.json();
    const { messages }: { messages: Message[] } = body ?? {};

    // 2) Validación mínima
    const lastMessage = messages && messages.length ? messages[messages.length - 1] : undefined;
    if (!messages || !lastMessage || lastMessage.role !== "user") {
      return NextResponse.json(
        { error: "messages are required in the request body and the last message must be from the user" },
        { status: 400 }
      );
    }

    // 3) Normalizamos historial (EXCLUYE el último mensaje del usuario)
    const allowed = new Set<ChatMessage["role"]>(["system", "user", "assistant", "tool"]);
    const prior = messages.slice(0, -1);

    const chatHistory: ChatMessage[] = prior
      .filter((m: any) => allowed.has(m.role))
      .map((m: any) => ({
        role: m.role as ChatMessage["role"],
        content:
          typeof m.content === "string"
            ? m.content
            : Array.isArray(m.content)
            ? m.content
                .map((part: any) =>
                  typeof part === "string"
                    ? part
                    : "text" in (part ?? {})
                    ? String(part.text)
                    : ""
                )
                .join(" ")
            : String(m.content ?? ""),
      }));

    // 4) Texto del último mensaje (usuario) normalizado (evita error con .map en ternarios)
    let userContent = "";
    const lmContent = (lastMessage as any).content;
    if (typeof lmContent === "string") {
      userContent = lmContent;
    } else if (Array.isArray(lmContent)) {
      userContent = (lmContent as any[])
        .map((part: any) =>
          typeof part === "string"
            ? part
            : "text" in (part ?? {})
            ? String(part.text)
            : ""
        )
        .join(" ");
    } else {
      userContent = String(lmContent ?? "");
    }

    // 5) Instanciamos OpenAI vía LlamaIndex con gpt-4o-mini
    //    (cast `as any` para que compile en TS aunque el tipo no lo liste aún)
    const llm = new OpenAI({
      model: "gpt-4o-mini" as any,
    });

    // 6) Construimos el motor de chat (tu RAG)
    const chatEngine = await createChatEngine(llm);

    // 7) System prompt al INICIO del historial: fidelidad al PDF
    chatHistory.unshift({
      role: "system",
      content:
        "Eres un asistente que responde únicamente con información contenida en los documentos proporcionados. " +
        "Si no encuentras la respuesta en ellos, responde exactamente: 'No encontré esa información en el documento.' " +
        "No inventes datos ni uses conocimiento externo.",
    });

    // 8) Llamada POSICIONAL (✅ la que necesita tu proyecto):
    //    (message, chatHistory, stream)
    const response = await (chatEngine as any).chat(userContent, chatHistory, true);

    // 9) Adaptamos a stream para el cliente (SSE)
    const stream = LlamaIndexStream(response);

    // 10) Respondemos en streaming
    return new StreamingTextResponse(stream);
  } catch (error) {
    console.error("[LlamaIndex]", error);
    return NextResponse.json(
      { error: (error as Error).message ?? "Internal Server Error" },
      { status: 500 }
    );
  }
}
