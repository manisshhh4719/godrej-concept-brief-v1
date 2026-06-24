const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

dotenv.config({ path: path.join(__dirname, '.env') });

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// ── Gold standards storage ─────────────────────
const GOLD_STANDARDS_FILE = 'gold_standards.json';

function loadGoldStandards() {
  if (fs.existsSync(GOLD_STANDARDS_FILE)) {
    return JSON.parse(fs.readFileSync(GOLD_STANDARDS_FILE, 'utf8'));
  }
  return {};
}

function saveGoldStandards(data) {
  fs.writeFileSync(GOLD_STANDARDS_FILE, JSON.stringify(data, null, 2));
}

// ── Similarity matching ────────────────────────
function calculateSimilarity(text1, text2) {
  const words1 = new Set(text1.toLowerCase().split(/\s+/));
  const words2 = new Set(text2.toLowerCase().split(/\s+/));
  const intersection = new Set([...words1].filter(w => words2.has(w)));
  const union = new Set([...words1, ...words2]);
  return intersection.size / union.size;
}

function findSimilarBrief(prompt, goldStandards, threshold) {
  let bestMatch = null;
  let bestScore = 0;
  for (const [key, value] of Object.entries(goldStandards)) {
    const score = calculateSimilarity(prompt, value.originalPrompt);
    if (score > bestScore) {
      bestScore = score;
      bestMatch = { key, value, score };
    }
  }
  if (bestMatch && bestScore >= threshold) return bestMatch;
  return null;
}

// ── Concept Genie System Prompt ────────────────
const CONCEPT_GENIE_PROMPT = `ROLE: Neutral FMCG insight strategist + concept architect. Two jobs: synthesize multi-source consumer inputs into sharp insights and build commercially powerful, test-ready concepts. No summaries. Interpret. Connect patterns. Surface tensions. Leadership-ready. No fluff. No italics. No em dashes.

You are running in CREATE MODE. The user will provide structured inputs. Your job is to generate a concept card output in strict JSON format.

CONSISTENCY RULES (NON-NEGOTIABLE):
- Always output the same concept for the same inputs. Be deterministic.
- Never vary structure, order, or format between runs.
- Output ONLY the JSON object below. No preamble. No explanation. No markdown. No code blocks.

OUTPUT FORMAT (strict JSON only, no deviation):
{
  "consumer_tension": "Max 12 words in consumer language as question or statement",
  "brand_product_name": "Brand and product name",
  "format_tag": "Max 5 words",
  "benefit_headline": "One bold promise. Max 8 words.",
  "proof_block_1": {
    "claim_label": "Max 4 words",
    "rtb_line": "Max 9 words - ingredient/technology/mechanism language",
    "icon": "simple icon description"
  },
  "proof_block_2": {
    "claim_label": "Max 4 words",
    "rtb_line": "Max 9 words - ingredient/technology/mechanism language",
    "icon": "simple icon description"
  },
  "proof_block_3": {
    "claim_label": "Max 4 words",
    "rtb_line": "Max 9 words - ingredient/technology/mechanism language",
    "icon": "simple icon description"
  },
  "price_badge": "Price if available or null",
  "image_prompt": "Detailed prompt for a 16:9 premium FMCG advertisement. LEFT PANEL (50%): Emotional lifestyle photo of consumer experiencing 'consumer_tension'. RIGHT PANEL (50%): Extremely dense, polished FMCG retail layout. Top: Overarching promise (max 2 lines), thin divider, 'benefit_headline' (largest bold uppercase). Brand Lockup: '[Brand Logo] brand_product_name | Product Descriptor', followed by a capsule-shaped product system strip. Middle 2x2 grid (no borders, tight spacing): Top-Left: Proof block 1 (premium circular icon, headline, short proof). Top-Right: Proof block 2 (icon, headline, short proof). Bottom-Left: Proof block 3 (icon, headline, short proof). Bottom-Right: Massive hero photorealistic packshot (dominates quadrant, studio lighting, water droplets, highly readable) with 'price_badge' beside it. Bottom: Full-width dark footer bar with 3 meaningful benefit icons. Do NOT use placeholder section labels. Zero empty whitespace."
}

COPY DISCIPLINE: Maximum 55 total words. Exactly 3 proof blocks. No repetition.
RTB STRENGTH RULE: Every claim must have ingredient/technology/mechanism proof.`;

// ── Evaluator Prompt ───────────────────────────
const EVALUATOR_PROMPT = `You are an FMCG concept card evaluator. Score the concept on a scale of 0-100.

Criteria:
1. Consumer Tension clarity (0-20): Sharp, consumer language, relatable?
2. Benefit Headline strength (0-20): Bold, clear, one message, max 8 words?
3. RTB quality (0-30): Mechanism-based? Proves the claim?
4. Copy discipline (0-20): Under 55 words? No repetition? No fluff?
5. Commercial potential (0-10): Understood in 3 seconds?

Output ONLY a JSON object. No explanation. No preamble. No markdown. No code blocks:
{
  "total_score": 85,
  "tension_score": 18,
  "headline_score": 17,
  "rtb_score": 25,
  "copy_score": 16,
  "commercial_score": 9,
  "key_strength": "One sentence on biggest strength",
  "key_weakness": "One sentence on biggest weakness"
}`;

// ── Gemini API call ────────────────────────────
async function callGemini(systemPrompt, userPrompt, temperature = 0.7, retries = 3) {
  const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await axios.post(url, {
        system_instruction: {
          parts: [{ text: systemPrompt }]
        },
        contents: [{
          parts: [{ text: userPrompt }]
        }],
        generationConfig: {
          temperature: temperature,
          maxOutputTokens: 8192,
          thinkingConfig: {
            thinkingBudget: 0
          }
        }
      });

      const text = response.data.candidates[0].content.parts[0].text;
      console.log('=== GEMINI RAW RESPONSE ===');
      console.log(text.substring(0, 300) + '...');
      console.log('===========================');

      const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON found in response');

      let jsonStr = jsonMatch[0];
      jsonStr = jsonStr
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        .replace(/\n/g, ' ')
        .replace(/,\s*}/g, '}')
        .replace(/,\s*]/g, ']');

      return JSON.parse(jsonStr);
    } catch (error) {
      if (attempt === retries) {
        console.error('Final attempt failed:', error.message);
        throw error;
      }
      console.log(`API Error (Attempt ${attempt}): ${error.message}. Retrying in ${attempt * 3} seconds...`);
      await new Promise(r => setTimeout(r, attempt * 3000));
    }
  }
}

// ── Generate one concept ───────────────────────
async function generateConcept(prompt, temperature = 0.7) {
  return await callGemini(CONCEPT_GENIE_PROMPT, prompt, temperature);
}

// ── Score one concept ──────────────────────────
async function scoreConcept(concept) {
  const simplePrompt = `Score this FMCG concept card 0-100. Output ONLY JSON, no markdown:
{"total_score":85,"tension_score":18,"headline_score":17,"rtb_score":25,"copy_score":16,"commercial_score":9,"key_strength":"one sentence","key_weakness":"one sentence"}

Concept: ${JSON.stringify(concept)}`;
  return await callGemini('You are an FMCG concept evaluator. Output only valid JSON.', simplePrompt, 0);
}

// ── Delay helper ───────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Main concept generation route ─────────────
app.post('/api/generate-concept', async (req, res) => {
  try {
    const { prompt, similarityThreshold = 0.7 } = req.body;

    console.log('=== NEW REQUEST ===');
    console.log('Threshold:', similarityThreshold);

    // Step 1 — Check cache
    const goldStandards = loadGoldStandards();
    const similarBrief = findSimilarBrief(prompt, goldStandards, similarityThreshold);

    if (similarBrief) {
      console.log(`Cache hit — similarity: ${similarBrief.score.toFixed(2)}`);
      return res.json({
        result: similarBrief.value.concept,
        score: similarBrief.value.score,
        evaluation: similarBrief.value.evaluation,
        fromCache: true,
        similarityScore: similarBrief.score
      });
    }

    // Step 2 — Generate 1 concept
    console.log('Generating 1 concept...');
    const concept = await generateConcept(prompt, 0.7);

    // Step 3 — Score the concept
    console.log('Scoring...');
    const score = await scoreConcept(concept);
    
    const best = { concept, score: score.total_score, evaluation: score };
    console.log(`Best score: ${best.score}/100`);

    // Step 5 — Save gold standard
    const key = `brief_${Date.now()}`;
    goldStandards[key] = {
      originalPrompt: prompt,
      concept: best.concept,
      score: best.score,
      evaluation: best.evaluation,
      createdAt: new Date().toISOString()
    };
    saveGoldStandards(goldStandards);

    res.json({
      result: best.concept,
      score: best.score,
      evaluation: best.evaluation,
      fromCache: false
    });

  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    res.status(500).json({ error: error.message || 'Failed to generate concept' });
  }
});

// ── Image generation route ─────────────────────
app.post('/api/generate-image', async (req, res) => {
  try {
    const { prompt } = req.body;
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    const finalPrompt = prompt + " Ensure the image is in 16:9 aspect ratio.";
    
    console.log("Attempting Gemini image generation...");
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=${GEMINI_API_KEY}`;
      const response = await axios.post(url, {
        contents: [{ parts: [{ text: finalPrompt }] }],
        generationConfig: { 
          responseModalities: ['IMAGE', 'TEXT']
        }
      });

      const imageData = response.data.candidates[0].content.parts.find(p => p.inlineData);
      if (!imageData) throw new Error('No image generated by Gemini');

      const base64Image = imageData.inlineData.data;
      const mimeType = imageData.inlineData.mimeType;
      return res.json({ imageUrl: `data:${mimeType};base64,${base64Image}` });

    } catch (geminiError) {
      console.log('Gemini failed, falling back to Hugging Face...');
      
      const HF_API_KEY = process.env.HF_API_KEY;
      const hfImageUrl = 'https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell';
      
      const hfResponse = await axios.post(hfImageUrl, 
        { inputs: finalPrompt }, 
        {
          headers: { 
            Authorization: `Bearer ${HF_API_KEY}`, 
            'Content-Type': 'application/json',
            'Accept': 'image/png'
          },
          responseType: 'arraybuffer'
        }
      );
      
      const base64Image = Buffer.from(hfResponse.data, 'binary').toString('base64');
      const mimeType = hfResponse.headers['content-type'] || 'image/png';
      
      return res.json({ imageUrl: `data:${mimeType};base64,${base64Image}` });
    }

  } catch (error) {
    console.error('Image generation completely failed:', error.response ? Buffer.from(error.response.data).toString() : error.message);
    res.status(503).json({ error: 'Image generation failed on both providers' });
  }
});

// ── Gold standards management ──────────────────
app.get('/api/gold-standards', (req, res) => {
  res.json(loadGoldStandards());
});

app.delete('/api/gold-standards/:key', (req, res) => {
  const goldStandards = loadGoldStandards();
  delete goldStandards[req.params.key];
  saveGoldStandards(goldStandards);
  res.json({ success: true });
});

// Delete a gold standard by matching its original prompt content (category + tension)
// Used when a recent brief is trashed from the browser, so it stops affecting similarity matching
app.post('/api/gold-standards/delete-by-content', (req, res) => {
  try {
    const { category, tension } = req.body;
    const goldStandards = loadGoldStandards();
    let removed = 0;
    for (const [key, value] of Object.entries(goldStandards)) {
      const op = value.originalPrompt || '';
      const catMatch = category && op.includes(category);
      const tenMatch = tension && op.includes(tension);
      // Require both to match so we don't delete unrelated briefs
      if (catMatch && tenMatch) {
        delete goldStandards[key];
        removed++;
      }
    }
    saveGoldStandards(goldStandards);
    res.json({ success: true, removed });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ── Start server ───────────────────────────────
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
