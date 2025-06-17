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
  // const message = `Separate the glued string: ${userPrompt}. Look for common English words. If a recognized English word is found, separate it. After the english word is found and separated, if the word contains any mispelling, correct it (eg. \"Hesllo\" -> \"Hello\", \"Hosdla\" -> \"Hola\" ) Otherwise, if a single letter can be isolated, separate that. If neither, break into individual letters. Examples: \"howareyou\" -> \"How are you\", \"acchello\" -> \"acc hello\", \"tzoo\" -> \"t zoo\", \"zoo\" -> \"zoo\", \"zo\" -> \"z o\". Give me only the separated string as response, only alphabets A to Z are allowed.`;
  const message = `
  You are a professional assistant that corrects AUSLAN fingerspelling outputs into clean, well-formatted English sentences typically used in professional video conferencing environments.

  Your task is:
  - Fix any spelling mistakes in the raw input.
  - Intelligently separate words.
  - Capitalize proper nouns and start of sentences.
  - Insert appropriate punctuation like full stops or question marks.
  - Make the final result sound like fluent and appropriate English for a professional meeting context.

  Here are some examples:

  Input: himinumiisasim  
  Output: Hi, my name is Asim.

  Input: goodmorningeveryone  
  Output: Good morning, everyone.

  Input: heyguyscanuseemyscreen  
  Output: Hey guys, can you see my screen?

  Input: howareyougoingtoday  
  Output: How are you going today?

  Input: iguesswecanstartnow  
  Output: I guess we can start now.

  Input: iloveaustralia  
  Output: I love Australia.

  Input: heycananyonehearclearly  
  Output: Hey, can anyone hear clearly?

  Input: imgoodthankyou  
  Output: I'm good, thank you.

  Input: yesiseeyourscreen  
  Output: Yes, I see your screen.

  Input: sorryicanthearyouproperly  
  Output: Sorry, I can't hear you properly.

  Input: heywhatsyourname  
  Output: Hey, what's your name?

  Input: heyasimgoodafternoon  
  Output: Hey Asim, good afternoon.

  Input: gdaymatehowhaveyoubeen  
  Output: G'day mate, how have you been?

  Input: goodthankshowaboutyou  
  Output: Good, thanks. How about you?

  Input: canwestartthemeeting  
  Output: Can we start the meeting?

  Input: hicanyouseemyscreennow  
  Output: Hi, can you see my screen now?

  Input: letsgetstarted  
  Output: Let's get started.

  Input: goodafternoonteam  
  Output: Good afternoon, team.

  Input: okayletswaitforothers  
  Output: Okay, let's wait for others.

  Input: helloeveryonehopeyoudoingwell  
  Output: Hello everyone, hope you're doing well.

  Input: thankyouallforjoining  
  Output: Thank you all for joining.

  Input: asimherefromdeloitte  
  Output: Asim here from Deloitte.

  Input: welcomeeveryone  
  Output: Welcome, everyone.

  Input: hiihavejoinedfrommelbourne  
  Output: Hi, I have joined from Melbourne.

  Input: letswrapupfortoday  
  Output: Let's wrap up for today.

  Input: himynameisasimhoyarutou  
  Output: Hi, my name is Asim. How are you?

  Input: HEGLLO
  Output: Hello.

  Input: WZOO
  Output: Zoo.

  Input: HESLLO
  Output: Hello.

  Input: SEFHSEOGIHSROBH
  Output: S, E, F, H, S, E, O, G, I, H, S, R, O, B, H

  Now, correct this input:
  Input: ${userPrompt}
  Output:`;
  
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
