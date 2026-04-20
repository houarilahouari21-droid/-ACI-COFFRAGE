import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

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
      
      if (provider === "google") {
        const geminiKey = process.env.GEMINI_API_KEY;
        if (!geminiKey) return res.status(500).json({ error: "GEMINI_API_KEY manquante sur le serveur." });
        
        const genAI = new GoogleGenerativeAI(geminiKey);
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

        const result = await model.generateContent([
          {
            inlineData: {
              data: base64,
              mimeType: mimeType
            }
          },
          {
            text: `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les informations sur les dalles et les poutres.
            Pour chaque élément trouvé, donne :
            1. Le nom unique identifiant (ex: DALLE D1, POUTRE P2).
            2. L'épaisseur ou profondeur brute en millimètres (mm).
            3. Le type : "DALLE" ou "POUTRE".
            
            Réponds UNIQUEMENT avec un tableau JSON valide.
            Exemple: [{"name": "Dalle 1", "thickness": 200, "type": "DALLE"}]
            Si tu ne trouves rien, renvoie un tableau vide [].`
          }
        ]);
        
        const response = await result.response;
        const text = response.text();
        // Extract JSON from potential markdown blocks
        const jsonMatch = text.match(/\[.*\]/s);
        const extracted = JSON.parse(jsonMatch ? jsonMatch[0] : text);
        res.json({ elements: extracted });

      } else if (provider === "groq") {
        const groqKey = process.env.GROQ_API_KEY;
        if (!groqKey) return res.status(500).json({ error: "GROQ_API_KEY manquante sur le serveur." });

        const groqRes = await fetch("https://api.groq.com/openai/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${groqKey}`
          },
          body: JSON.stringify({
            model: "llama-3.2-11b-vision-preview",
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    text: `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les informations sur les dalles et les poutres.
                    IMPORTANT: Tu dois retourner UNIQUEMENT un objet JSON avec une clé "elements" contenant la liste des objets.
                    Chaque objet doit avoir: "name" (string), "thickness" (number, mm), "type" (DALLE ou POUTRE).
                    
                    Exemple de format attendu:
                    { "elements": [{"name": "Dalle 1", "thickness": 200, "type": "DALLE"}] }
                    Si tu ne trouves rien, renvoie { "elements": [] }.`
                  },
                  {
                    type: "image_url",
                    image_url: {
                      url: `data:${mimeType};base64,${base64}`
                    }
                  }
                ]
              }
            ],
            response_format: { type: "json_object" }
          })
        });

        if (!groqRes.ok) {
          const errData = await groqRes.json();
          return res.status(groqRes.status).json({ error: errData.error?.message || "Erreur Groq" });
        }

        const data = await groqRes.json();
        const content = data.choices[0].message.content;
        const parsed = JSON.parse(content);
        res.json({ elements: parsed.elements || [] });
      } else {
        res.status(400).json({ error: "Provider inconnu" });
      }
    } catch (error: any) {
      console.error("Extraction error:", error);
      res.status(500).json({ error: error.message });
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
