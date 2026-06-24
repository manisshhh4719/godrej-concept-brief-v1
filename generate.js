const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const axios = require('axios');

dotenv.config({ path: path.join(__dirname, '.env') });
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const HF_API_KEY = process.env.HF_API_KEY;

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

const userPrompt = `Stage 1 - Category: Personal Care > Hair Care. Context: Shampoo category in India is dominated by dandruff and damage repair. Anti-hairfall is the fastest growing sub-segment but underpenetrated in mass market. Stage 2 - Market: India, Tier 1 and Tier 2 cities. Young working professionals, nuclear families. Rising screen time and stress levels driving hair health concerns. Mass-premium price band (Rs 150-350) is the sweet spot. Stage 3 - Target Group: Men and women aged 22-35. Urban, working. Noticing more hair in the drain every morning. Stressed, sleep-deprived, eating poorly. Tried multiple shampoos but nothing feels like it actually works. Stage 4 - Consumer Tension: I see more hair falling every day and I don't know if my shampoo is helping or making it worse. I just want something that actually stops the loss. Stage 5 - Unmet Need: No mass-market shampoo credibly addresses hairfall with visible proof of results. Either it's a medical-looking bottle that feels clinical, or it's a pretty bottle with no real promise. Stage 6 - JTBD: Help me feel confident that I am actively doing something every wash to stop my hair from thinning. Stage 7 - Product Details: Working name: Godrej RootLock Shampoo. 340ml bottle at Rs 249. Refill pouch at Rs 179. Contains Redensyl + Biotin + Caffeine complex. Stage 8 - Alternatives: Dove Hairfall Rescue, Indulekha, Clinic Plus, Mamaearth.`;

async function generateText() {
  try {
    console.log('Attempting to generate text with Gemini...');
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;
    const response = await axios.post(url, {
      system_instruction: { parts: [{ text: CONCEPT_GENIE_PROMPT }] },
      contents: [{ parts: [{ text: userPrompt }] }],
      generationConfig: { temperature: 0.7, maxOutputTokens: 8192 }
    });
    return response.data.candidates[0].content.parts[0].text;
  } catch (err) {
    console.log('Gemini text generation failed. Falling back to Hugging Face...');
    const hfUrl = 'https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions';
    const response = await axios.post(hfUrl, {
      model: 'meta-llama/Llama-3.3-70B-Instruct',
      messages: [
        { role: 'system', content: CONCEPT_GENIE_PROMPT },
        { role: 'user', content: userPrompt }
      ],
      max_tokens: 2000,
      temperature: 0.7
    }, {
      headers: { Authorization: `Bearer ${HF_API_KEY}`, 'Content-Type': 'application/json' }
    });
    return response.data.choices[0].message.content;
  }
}

async function generateImage(prompt) {
  try {
    console.log('Attempting to generate image with Gemini...');
    const finalImagePrompt = prompt + " Ensure the image is in 16:9 aspect ratio.";
    const imageUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent?key=${GEMINI_API_KEY}`;
    const imgResponse = await axios.post(imageUrl, {
      contents: [{ parts: [{ text: finalImagePrompt }] }],
      generationConfig: {
        responseModalities: ['IMAGE', 'TEXT']
      }
    });
    const imageData = imgResponse.data.candidates[0].content.parts.find(p => p.inlineData);
    if (!imageData) throw new Error('No image generated by Gemini');
    return Buffer.from(imageData.inlineData.data, 'base64');
  } catch (err) {
    console.log('Gemini image generation failed. Falling back to Hugging Face...');
    const finalImagePrompt = prompt + " Ensure the image is in 16:9 aspect ratio.";
    const hfImageUrl = 'https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell';
    try {
      const hfResponse = await axios.post(hfImageUrl, {
        inputs: finalImagePrompt
      }, {
        headers: { 
          Authorization: `Bearer ${HF_API_KEY}`, 
          'Content-Type': 'application/json',
          'Accept': 'image/png'
        },
        responseType: 'arraybuffer'
      });
      return Buffer.from(hfResponse.data);
    } catch (hfErr) {
      if (hfErr.response && hfErr.response.data) {
        throw new Error("HF Image Error: " + Buffer.from(hfErr.response.data).toString());
      }
      throw hfErr;
    }
  }
}

async function main() {
  try {
    console.log('1. Generating Concept Card from text...');
    const text = await generateText();
    
    const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON found in response');

    const concept = JSON.parse(jsonMatch[0]);
    console.log('\n--- Generated Concept ---');
    console.log(JSON.stringify(concept, null, 2));
    console.log('-------------------------\n');

    const imagePrompt = concept.image_prompt;
    if (!imagePrompt) {
      console.log('No image_prompt returned from the model.');
      return;
    }

    console.log('2. Generating Image from image_prompt...');
    console.log('Prompt: ' + imagePrompt);
    
    const imageBuffer = await generateImage(imagePrompt);

    // Create image directory
    const imgDir = path.join(__dirname, 'image');
    if (!fs.existsSync(imgDir)) {
      fs.mkdirSync(imgDir);
    }

    const filePath = path.join(imgDir, 'concept_hero.png');
    fs.writeFileSync(filePath, imageBuffer);
    console.log(`\nSuccess! Image saved to: ${filePath}`);

  } catch (err) {
    console.error('Error:', err.message);
    if (err.response) {
      console.error(err.response.data);
    }
  }
}

main();
