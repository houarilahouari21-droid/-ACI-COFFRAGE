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
