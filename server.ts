import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import Groq from "groq-sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";

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

      if (provider === "google") {
        const geminiKey = process.env.GEMINI_API_KEY;
        if (!geminiKey) return res.status(500).json({ error: "Clé API Gemini (GEMINI_API_KEY) manquante sur le serveur." });

        const genAI = new GoogleGenerativeAI(geminiKey);
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });

        const prompt = `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les informations sur les dalles et les poutres.
        Retourne UNIQUEMENT un objet JSON: { "elements": [ { "name": string, "thickness": number, "type": "DALLE"|"POUTRE" } ] }.`;

        const result = await model.generateContent([
          prompt,
          {
            inlineData: {
              data: base64,
              mimeType: mimeType || "image/jpeg"
            }
          }
        ]);

        const response = await result.response;
        let text = response.text();
        
        // Nettoyage Markdown si nécessaire
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        const parsed = JSON.parse(jsonMatch ? jsonMatch[0] : text);
        return res.json({ elements: parsed.elements || [] });

      } else if (provider === "groq") {
        const groqKey = process.env.GROQ_API_KEY || process.env.VITE_GROQ_API_KEY;
        if (!groqKey) return res.status(500).json({ error: "GROQ_API_KEY manquante sur le serveur." });

        const groq = new Groq({ apiKey: groqKey });

        const data = await groq.chat.completions.create({
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
          model: "llama-3.2-11b-vision-preview",
          response_format: { type: "json_object" }
        });

        const content = data.choices[0].message.content;
        if (!content) {
          console.error("Groq Empty Response:", JSON.stringify(data));
          throw new Error("Aucune réponse reçue de Groq.");
        }

        try {
          const jsonMatch = content.match(/\{[\s\S]*\}/);
          const parsed = JSON.parse(jsonMatch ? jsonMatch[0] : content);
          res.json({ elements: parsed.elements || [] });
        } catch (e) {
          console.error("Groq JSON Parse Error:", content);
          throw new Error("Groq n'a pas retourné un JSON valide.");
        }
      } else if (provider === "openrouter") {
        const orKey = process.env.OPENROUTER_API_KEY;
        if (!orKey) return res.status(500).json({ error: "OPENROUTER_API_KEY manquante sur le serveur." });

        // On utilise les modèles envoyés par le client ou une liste par défaut stable
        let modelIds: string[] = req.body.models || [
          "anthropic/claude-3.5-sonnet",
          "google/gemini-pro-1.5",
          "openai/gpt-4o-mini"
        ];

        // Mapping simple pour les noms d'affichage
        const getDisplayName = (id: string) => {
          if (id.includes("claude-3.5")) return "Claude 3.5 Sonnet";
          if (id.includes("gemini-2.0")) return "Gemini 2.0 Flash";
          if (id.includes("gemini-pro")) return "Gemini 1.5 Pro";
          if (id.includes("gpt-4o-mini")) return "GPT-4o Mini";
          if (id.includes("gpt-4o")) return "GPT-4o";
          if (id.includes("llama")) return id.includes("90b") ? "Llama 3.2 90B" : "Llama 3.2 11B";
          if (id.includes("grok")) return "Grok 2 Vision";
          if (id.includes("qwen")) return "Qwen 2 VL";
          if (id.includes("pixtral")) return "Pixtral 12B";
          return id.split("/")[1]?.toUpperCase() || id;
        };

        const executeExtraction = async (ids: string[]) => {
          return await Promise.allSettled(ids.map(async (modelId) => {
            const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
              method: "POST",
              headers: {
                "Authorization": `Bearer ${orKey}`,
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Coffrage AI Assistant",
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                model: modelId,
                messages: [
                  {
                    role: "user",
                    content: [
                      {
                        type: "text",
                        text: `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les dalles et poutres.
                        Retourne UNIQUEMENT un objet JSON: { "elements": [ { "name": string, "thickness": number, "type": "DALLE"|"POUTRE" } ] }.`
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
              throw new Error(err.error?.message || `Status ${response.status}`);
            }

            const data = await response.json();
            const content = data.choices?.[0]?.message?.content;
            if (!content) throw new Error("Réponse vide");

            const jsonMatch = content.match(/\{[\s\S]*\}/);
            const parsed = JSON.parse(jsonMatch ? jsonMatch[0] : content);
            return {
              modelName: getDisplayName(modelId),
              modelId: modelId,
              elements: parsed.elements || []
            };
          }));
        };

        let results = await executeExtraction(modelIds);
        let candidates = results
          .filter((r): r is PromiseFulfilledResult<any> => r.status === 'fulfilled')
          .map(r => r.value)
          .filter(c => c.elements.length > 0);

        // Si ÉCHEC TOTAL (No endpoints found, etc.), on tente une dernière fois avec un modèle ultra-stable
        if (candidates.length === 0) {
          console.warn("Échec OpenRouter avec les modèles sélectionnés, tentative de secours...");
          const retryResults = await executeExtraction(["openai/gpt-4o-mini", "google/gemini-flash-1.5"]);
          candidates = retryResults
            .filter((r): r is PromiseFulfilledResult<any> => r.status === 'fulfilled')
            .map(r => r.value)
            .filter(c => c.elements.length > 0);
            
          if (candidates.length === 0) {
            const errors = results
              .filter((r): r is PromiseRejectedResult => r.status === 'rejected')
              .map(r => r.reason.message)
              .join(" | ");
            throw new Error(`Tous les modèles ont échoué. Vérifiez vos crédits OpenRouter ou essayez plus tard. Erreurs : ${errors}`);
          }
        }

        return res.json({ candidates });

      } else if (provider === "huggingface") {
        const hfKey = process.env.HUGGINGFACE_API_KEY;
        if (!hfKey) return res.status(500).json({ error: "HUGGINGFACE_API_KEY manquante sur le serveur." });

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
                  { type: "text", text: "Extract structural elements (DALLE or POUTRE) from this image as JSON: { 'elements': [ { name, thickness, type } ] }." },
                  { type: "image_url", image_url: { url: `data:${mimeType || "image/jpeg"};base64,${base64}` } }
                ]
              }
            ],
            max_tokens: 1000
          })
        });

        if (!chatResponse.ok) {
          const err = await chatResponse.json().catch(() => ({}));
          throw new Error(err.error?.message || err.error || `Hugging Face Error: ${chatResponse.status}`);
        }

        const data = await chatResponse.json();
        const content = data.choices?.[0]?.message?.content;
        if (!content) {
          console.error("HF Response Body:", JSON.stringify(data));
          throw new Error("Aucune réponse de Hugging Face (Modèle peut-être en cours de chargement).");
        }

        try {
          const jsonMatch = content.match(/\{[\s\S]*\}/);
          const parsed = JSON.parse(jsonMatch ? jsonMatch[0] : content);
          res.json({ elements: parsed.elements || [] });
        } catch (e) {
          console.error("JSON Parse Error (HF):", content);
          throw new Error("Le modèle Hugging Face n'a pas produit de JSON exploitable.");
        }

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
