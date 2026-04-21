import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import Groq from "groq-sdk";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "50mb" }));

  // API Routes
  app.post("/api/ai-extract", async (req, res) => {
    try {
      const { base64, mimeType, provider } = req.body;
      
      if (!base64) return res.status(400).json({ error: "Image manquante" });

      if (provider === "groq") {
        const groqKey = process.env.GROQ_API_KEY || process.env.VITE_GROQ_API_KEY;
        if (!groqKey) return res.status(500).json({ error: "GROQ_API_KEY manquante sur le serveur. Configurez-la dans les paramètres." });

        const groq = new Groq({ apiKey: groqKey });

        const completion = await groq.chat.completions.create({
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les informations sur les dalles et les poutres.
                  IMPORTANT: Tu dois retourner UNIQUEMENT un objet JSON avec une clé "elements" contenant la liste des objets.
                  Chaque objet doit avoir: "name" (string), "thickness" (number, mm), "type" (DALLE ou POUTRE).
                  Si tu ne trouves rien, renvoie { "elements": [] }.`
                },
                {
                  type: "image_url",
                  image_url: {
                    url: `data:${mimeType || "image/jpeg"};base64,${base64}`
                  }
                }
              ]
            }
          ],
          model: "llama-3.3-70b-versatile",
          response_format: { type: "json_object" }
        });

        const content = completion.choices[0].message.content;
        if (!content) throw new Error("Aucune réponse de l'IA.");

        const parsed = JSON.parse(content);
        res.json({ elements: parsed.elements || [] });
      } else if (provider === "openrouter") {
        const orKey = process.env.OPENROUTER_API_KEY;
        if (!orKey) return res.status(500).json({ error: "OPENROUTER_API_KEY manquante sur le serveur." });

        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${orKey}`,
            "HTTP-Referer": process.env.APP_URL || "https://ai.studio/",
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: "google/gemini-flash-1.5", // Good balance for vision tasks on OpenRouter
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    text: `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les informations sur les dalles et les poutres.
                    IMPORTANT: Tu dois retourner UNIQUEMENT un objet JSON avec une clé "elements" contenant la liste des objets.
                    Chaque objet doit avoir: "name" (string), "thickness" (number, mm), "type" (DALLE ou POUTRE).`
                  },
                  {
                    type: "image_url",
                    image_url: {
                      url: `data:${mimeType || "image/jpeg"};base64,${base64}`
                    }
                  }
                ]
              }
            ],
            response_format: { type: "json_object" }
          })
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.error?.message || `OpenRouter Error: ${response.status}`);
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;
        if (!content) throw new Error("Aucune réponse d'OpenRouter.");
        
        const parsed = JSON.parse(content);
        res.json({ elements: parsed.elements || [] });

      } else if (provider === "huggingface") {
        const hfKey = process.env.HUGGINGFACE_API_KEY;
        if (!hfKey) return res.status(500).json({ error: "HUGGINGFACE_API_KEY manquante sur le serveur." });

        // Using Qwen2-VL-7B via Inference API (requires Pro or specific endpoints usually, but we'll try a common one)
        // Note: Vision API on HF Inference API can be tricky with base64 directly in messages sometimes depending on model
        const response = await fetch("https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${hfKey}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            inputs: `data:${mimeType || "image/jpeg"};base64,${base64}`,
            parameters: {
              max_new_tokens: 1024,
              return_full_text: false,
            },
            // Note: This is simplified. Some vision models on HF expect specific task signatures.
            // But we'll try the common chat-like interface if the model supports it.
            // Realistically, for structured extraction, Gemini/Groq/OR are safer.
          })
        });

        // Better approach for HF: many models use the chat completion API too now
        const chatResponse = await fetch("https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct/v1/chat/completions", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${hfKey}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            model: "Qwen/Qwen2-VL-7B-Instruct",
            messages: [
              {
                role: "user",
                content: [
                  { type: "text", text: "Extract structural elements (DALLE or POUTRE) from this image as JSON with an 'elements' array containing {name, thickness, type}." },
                  { type: "image_url", image_url: { url: `data:${mimeType || "image/jpeg"};base64,${base64}` } }
                ]
              }
            ],
            max_tokens: 1000
          })
        });

        if (!chatResponse.ok) {
          const err = await chatResponse.json().catch(() => ({}));
          throw new Error(err.error || `Hugging Face Error: ${chatResponse.status}`);
        }

        const data = await chatResponse.json();
        const content = data.choices?.[0]?.message?.content;
        // Clean markdown if present
        const jsonMatch = content?.match(/\{[\s\S]*\}/);
        const cleanContent = jsonMatch ? jsonMatch[0] : content;
        
        const parsed = JSON.parse(cleanContent);
        res.json({ elements: parsed.elements || [] });

      } else {
        res.status(400).json({ error: "Le provider '" + provider + "' doit être géré côté client (Gemini) ou n'est pas supporté." });
      }
    } catch (error: any) {
      console.error("Extraction error:", error.message);
      res.status(500).json({ error: error.message || "Erreur interne du serveur" });
    }
  });

  // Vite setup
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
