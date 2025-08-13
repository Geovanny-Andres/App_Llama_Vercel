// app/api/chat/route.ts
// API route para manejar el chat con LlamaIndex desde Next.js (App Router)

import { Message, StreamingTextResponse } from "ai";
import { OpenAI } from "llamaindex";
import { NextRequest, NextResponse } from "next/server";
import { createChatEngine } from "./engine";
import { LlamaIndexStream } from "./llamaindex-stream";

// Tipo que espera el motor de LlamaIndex para el historial
type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
};

// Forzamos runtime Node y desactivamos caché (respuestas dinámicas)
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    // 1) Leemos el body y extraemos mensajes
    const body = await request.json();
    const { messages }: { messages: Message[] } = body ?? {};

    // 2) Último mensaje (usuario)
    const lastMessage = messages && messages.length ? messages[messages.length - 1] : undefined;
    if (!messages || !lastMessage || lastMessage.role !== "user") {
      return NextResponse.json(
        { error: "messages are required in the request body and the last message must be from the user" },
        { status: 400 }
      );
    }

    // 3) Historial (sin el último mensaje) filtrando roles soportados
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

    // 4) Contenido del usuario normalizado
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

    // 5) Instanciamos el LLM
    const llm = new OpenAI({
      model: "gpt-3.5-turbo", // Cambia si quieres otro modelo
    });

    // 6) Creamos el motor de chat
    const chatEngine = await createChatEngine(llm);

    // 7) Ejecutamos el chat
    // ⚠️ Aquí está el cambio clave: usamos SIEMPRE la firma (message, chatHistory, stream)
    // Esto evita el error "chatHistory.push is not a function"
    const response = await (chatEngine as any).chat(userContent, chatHistory, true);

    // 8) Convertimos la respuesta en un ReadableStream para el cliente (SSE/streaming)
    const stream = LlamaIndexStream(response);

    // 9) Enviamos la respuesta en streaming
    return new StreamingTextResponse(stream);
  } catch (error) {
    // Manejo de errores
    console.error("[LlamaIndex]", error);
    return NextResponse.json(
      { error: (error as Error).message ?? "Internal Server Error" },
      { status: 500 }
    );
  }
}
