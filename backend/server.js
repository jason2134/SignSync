const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config(); // Loads .env

const app = express();
app.use(bodyParser.json());
app.use(cors());
const PORT = 5001;
const GEMINI_KEY = process.env.GEMINI_API_KEY;

// Proxy endpoint to Gemini
app.post('/ask', async (req, res) => {
  const userPrompt = req.body.prompt;
  const message = `Separate the glued string: ${userPrompt}. Look for common English words. If a recognized English word is found, separate it. After the english word is found and separated, if the word contains any mispelling, correct it (eg. \"Hesllo\" -> \"Hello\", \"Hosdla\" -> \"Hola\" ) Otherwise, if a single letter can be isolated, separate that. If neither, break into individual letters. Examples: \"howareyou\" -> \"How are you\", \"acchello\" -> \"acc hello\", \"tzoo\" -> \"t zoo\", \"zoo\" -> \"zoo\", \"zo\" -> \"z o\". Give me only the separated string as response, only alphabets A to Z are allowed.`;
  try {
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_KEY}`,
      {
        contents: [
          {
            parts: [
              {
                text: message
              }
            ]
          }
        ]
      },
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error('Error calling Gemini API:', error.message);
    res.status(500).json({ error: 'Failed to get response from Gemini' });
  }
});

app.listen(PORT, () => {
  console.log(`Gemini backend server running at http://localhost:${PORT}`);
});
