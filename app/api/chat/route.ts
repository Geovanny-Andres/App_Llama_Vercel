// API route para manejar el chat con LlamaIndex desde Next.js (App Router)

import { Message, StreamingTextResponse } from "ai";
import type { ChatMessage } from "llamaindex"; // <-- el tipo correcto viene de llamaindex
import { OpenAI } from "llamaindex";
import { NextRequest, NextResponse } from "next/server";
import { createChatEngine } from "./engine";
import { LlamaIndexStream } from "./llamaindex-stream";

// Forzamos runtime Node y desactivamos caché (respuestas dinámicas)
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    // 1) Leemos el body y extraemos los mensajes del historial
    const body = await request.json();
    const { messages }: { messages: Message[] } = body;

    // 2) Tomamos el último mensaje (debe ser del usuario)
    const lastMessage = messages?.pop();
    if (!messages || !lastMessage || lastMessage.role !== "user") {
      return NextResponse.json(
        {
          error:
            "messages are required in the request body and the last message must be from the user",
        },
        { status: 400 },
      );
    }

    // 3) Normalizamos el historial de Message[] → ChatMessage[]
    //    (el tipo Message puede traer roles como "data"/"function" que NO acepta ChatMessage)
    const allowedRoles = new Set<ChatMessage["role"]>([
      "system",
      "user",
      "assistant",
      "tool",
    ]);

    const chatHistory: ChatMessage[] = messages
      .filter((m: any) => allowedRoles.has(m.role)) // filtramos roles no soportados
      .map((m: any) => ({
        role: m.role as ChatMessage["role"], // tipamos al conjunto de roles válidos
        content: m.content,                  // mantenemos el contenido tal cual
      }));

    // 4) Instanciamos el LLM (usa OPENAI_API_KEY desde variables de entorno)
    const llm = new OpenAI({
      model: "gpt-3.5-turbo",
    });

    // 5) Construimos el motor de chat (tu RAG con el storage generado)
    const chatEngine = await createChatEngine(llm);

    // 6) Ejecutamos el chat pasando:
    //    - el contenido del último mensaje del usuario
    //    - el historial ya normalizado (ChatMessage[])
    //    - true para habilitar streaming (según tu implementación)
    const response = await chatEngine.chat(
      lastMessage.content,
      chatHistory,
      true,
    );

    // 7) Transformamos la respuesta a un ReadableStream para el cliente
    const stream = LlamaIndexStream(response);

    // 8) Devolvemos la respuesta en streaming
    return new StreamingTextResponse(stream);
  } catch (error) {
    // Manejo de errores
    console.error("[LlamaIndex]", error);
    return NextResponse.json(
      { error: (error as Error).message },
      { status: 500 },
    );
  }
}
