import os
import json
import warnings
import textwrap
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
AUDIO_DIR = Path("sample_audio")
AUDIO_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

WHISPER_MODEL_SIZE = "base"

SAMPLE_CALLS = [
    {
        "id": "CALL-001",
        "filename": "call_001_closed.wav",
        "expected_category": "Closed",
        "script": (
            "Hello, I called last week about my Smartnode touch switch not "
            "responding to the app. Your technician visited yesterday and "
            "replaced the Wi-Fi module. Everything is working perfectly now. "
            "The lights respond instantly and the schedule feature is back. "
            "I just wanted to confirm the issue is fully resolved. Thank you "
            "for the quick support!"
        ),
    },
    {
        "id": "CALL-002",
        "filename": "call_002_open.wav",
        "expected_category": "Open",
        "script": (
            "Hi, I am a dealer from Ahmedabad. I installed Smartnode curtain "
            "controllers in a client's villa last month. Two of the five units "
            "are showing intermittent connectivity drops with the hub. "
            "The curtains sometimes move on their own. I have already tried "
            "re-pairing them but the problem comes back after a day. "
            "Could you please look into this and let me know what firmware "
            "update or replacement is needed? I am waiting for your response."
        ),
    },
    {
        "id": "CALL-003",
        "filename": "call_003_urgent.wav",
        "expected_category": "Urgent",
        "script": (
            "This is an emergency! The Smartnode smart door lock at my home "
            "has completely stopped working. We cannot unlock the front door "
            "and my elderly parents are locked inside. The battery shows full "
            "charge in the app but the motor does not engage at all. "
            "We need a technician immediately — this is a safety issue! "
            "Please escalate this right now, we cannot wait until tomorrow."
        ),
    },
    {
        "id": "CALL-004",
        "filename": "call_004_closed.wav",
        "expected_category": "Closed",
        "script": (
            "Good afternoon. I previously raised a complaint about the IR "
            "blaster not controlling my air conditioner. The support team "
            "guided me through the re-learning process over a video call "
            "and now it works perfectly with all three ACs in my home. "
            "The issue stands resolved and I am very satisfied with the "
            "service. You can close this ticket. Thanks a lot."
        ),
    },
    {
        "id": "CALL-005",
        "filename": "call_005_urgent.wav",
        "expected_category": "Urgent",
        "script": (
            "I am calling from a hotel in Surat that uses Smartnode "
            "automation across 50 rooms. Since this morning, the entire "
            "system is down — none of the room panels are responding, lights "
            "and ACs cannot be controlled. We have guests checking in and "
            "this is causing serious business impact. We need your engineering "
            "team on-site today. This is extremely critical and urgent."
        ),
    },
    {
        "id": "CALL-006",
        "filename": "call_006_open.wav",
        "expected_category": "Open",
        "script": (
            "Hello, I bought the Smartnode scene controller last week. "
            "I am trying to set up a 'Good Night' scene that turns off all "
            "lights and sets the AC to 24 degrees, but the app keeps showing "
            "an error when I try to save the scene. I have the latest app "
            "version. Could your team please check if this is a known bug "
            "and provide a timeline for the fix? No rush, but I would like "
            "an update when possible."
        ),
    },
]

def generate_audio_samples():
    """
    Converts text scripts to .mp3 audio files using Google Text-to-Speech.
    gTTS saves MP3 natively — no ffmpeg or pydub needed.
    Whisper can read MP3 files directly.
    """
    try:
        from gtts import gTTS

        print("[AUDIO] Generating synthetic audio samples with gTTS...\n")
        for call in SAMPLE_CALLS:
            
            mp3_name = Path(call["filename"]).with_suffix(".mp3").name
            filepath = AUDIO_DIR / mp3_name
            if filepath.exists():
                print(f"   + {mp3_name} already exists, skipping.")
                continue

            tts = gTTS(text=call["script"], lang="en", slow=False)
            tts.save(str(filepath))
            print(f"   + Generated {mp3_name}")

        print("\n[OK] Audio generation complete.\n")
        return True

    except ImportError:
        print("[WARN] gTTS not installed. Using transcripts directly.\n")
        return False
    except Exception as e:
        print(f"[WARN] Audio generation failed: {e}")
        print("   Falling back to text scripts.\n")
        return False
def transcribe_audio(audio_path: str, model=None) -> str:
    """
    Transcribe a single audio file using Whisper.
    Returns the transcription text.
    """
    result = model.transcribe(str(audio_path), language="en", fp16=False)
    return result["text"].strip()


def transcribe_all_calls(use_audio: bool) -> list[dict]:
    """
    Transcribe all sample calls.
    If audio files are available, uses Whisper; otherwise uses the scripts.
    """
    results = []

    if use_audio:
        try:
            import whisper

            print(f"[TRANSCRIBE] Loading Whisper model ({WHISPER_MODEL_SIZE})...")
            model = whisper.load_model(WHISPER_MODEL_SIZE)
            print("   + Model loaded.\n")

            for call in SAMPLE_CALLS:
                mp3_name = Path(call["filename"]).with_suffix(".mp3").name
                filepath = AUDIO_DIR / mp3_name
                print(f"   Transcribing {call['id']}... ", end="", flush=True)
                transcript = transcribe_audio(filepath, model)
                print("done.")
                results.append({
                    "call_id": call["id"],
                    "filename": call["filename"],
                    "expected_category": call["expected_category"],
                    "transcript": transcript,
                    "source": "whisper",
                })
            print()
            return results

        except ImportError:
            print("[WARN] Whisper not installed. Falling back to scripts.\n")
        except (FileNotFoundError, OSError) as e:
            print(f"\n[WARN] Whisper requires ffmpeg to decode audio: {e}")
            print("       Install ffmpeg: winget install ffmpeg")
            print("       Falling back to text scripts.\n")
        except Exception as e:
            print(f"\n[WARN] Transcription failed: {e}")
            print("       Falling back to text scripts.\n")
    print("[TRANSCRIBE] Using raw scripts as transcripts (audio not available).\n")
    for call in SAMPLE_CALLS:
        results.append({
            "call_id": call["id"],
            "filename": call["filename"],
            "expected_category": call["expected_category"],
            "transcript": call["script"],
            "source": "script_fallback",
        })
    return results
URGENT_KEYWORDS = [
    "emergency", "urgent", "immediately", "locked", "safety",
    "cannot wait", "critical", "dangerous", "fire", "stuck",
    "escalate", "right now", "on-site today", "extremely critical",
    "system is down", "business impact", "not working at all",
    "locked inside",
]

CLOSED_KEYWORDS = [
    "resolved", "working perfectly", "close this ticket", "fixed",
    "fully resolved", "issue is resolved", "satisfied", "working fine",
    "thank you for the support", "stands resolved", "works perfectly",
    "everything is working", "no further action",
]

OPEN_KEYWORDS = [
    "waiting for your response", "let me know", "could you please",
    "looking into", "pending", "timeline for the fix", "when possible",
    "please check", "need an update", "tried but", "problem comes back",
    "intermittent", "no rush",
]


def classify_rule_based(transcript: str) -> dict:
    """
    Classify a transcript using keyword/phrase matching.
    Returns category and matched evidence.
    """
    text_lower = transcript.lower()

    urgent_hits = [kw for kw in URGENT_KEYWORDS if kw in text_lower]
    closed_hits = [kw for kw in CLOSED_KEYWORDS if kw in text_lower]
    open_hits   = [kw for kw in OPEN_KEYWORDS if kw in text_lower]

    scores = {
        "Urgent": len(urgent_hits) * 2,
        "Closed": len(closed_hits),
        "Open":   len(open_hits),
    }

    predicted = max(scores, key=scores.get)
    if all(v == 0 for v in scores.values()):
        predicted = "Open"  

    return {
        "predicted_category": predicted,
        "confidence_note": f"Matched keywords — Urgent:{len(urgent_hits)}, "
                           f"Closed:{len(closed_hits)}, Open:{len(open_hits)}",
        "evidence": {
            "urgent_matches": urgent_hits,
            "closed_matches": closed_hits,
            "open_matches": open_hits,
        },
    }

def classify_zero_shot(transcript: str, classifier=None) -> dict:
    """
    Classify transcript using HuggingFace zero-shot classification.
    """
    candidate_labels = [
        "Closed — issue resolved, no action needed",
        "Open — issue acknowledged, pending resolution",
        "Urgent — critical issue requiring immediate escalation",
    ]

    result = classifier(transcript, candidate_labels)

    label_map = {
        "Closed — issue resolved, no action needed": "Closed",
        "Open — issue acknowledged, pending resolution": "Open",
        "Urgent — critical issue requiring immediate escalation": "Urgent",
    }

    scores_mapped = {}
    for label, score in zip(result["labels"], result["scores"]):
        cat = label_map.get(label, label)
        scores_mapped[cat] = round(score, 4)

    predicted = max(scores_mapped, key=scores_mapped.get)

    return {
        "predicted_category": predicted,
        "confidence": scores_mapped[predicted],
        "all_scores": scores_mapped,
    }


def classify_all_zero_shot(transcripts: list[dict]) -> list[dict]:
    """
    Run zero-shot classification on all transcripts.
    Falls back to rule-based if transformers is not available.
    """
    try:
        from transformers import pipeline

        print("[ZERO-SHOT] Loading zero-shot classification model...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1, 
        )
        print("   + Model loaded.\n")

        results = []
        for t in transcripts:
            print(f"   Classifying {t['call_id']}... ", end="", flush=True)
            zs_result = classify_zero_shot(t["transcript"], classifier)
            print(f"{zs_result['predicted_category']} "
                  f"(conf: {zs_result['confidence']:.2%})")
            results.append({**t, "zero_shot": zs_result})

        print()
        return results

    except ImportError:
        print("[WARN] transformers not installed. Skipping zero-shot.\n")
        return [{**t, "zero_shot": None} for t in transcripts]

def display_results(classified_data: list[dict]):
    """Pretty-print the classification results."""

    separator = "=" * 80

    print(separator)
    print("  SMARTNODE CUSTOMER SUPPORT -- CALL CLASSIFICATION RESULTS")
    print(separator)
    print()

    for item in classified_data:
        rule_result = classify_rule_based(item["transcript"])
        zs_result   = item.get("zero_shot")

        print(f"+-- {item['call_id']}  ({item['filename']})")
        print(f"|   Expected Category  : {item['expected_category']}")
        print(f"|   Transcript Source   : {item['source']}")
        print(f"|")
        print(f"|   [TRANSCRIPT] (first 200 chars):")
        wrapped = textwrap.fill(item["transcript"][:200] + "...", width=65)
        for line in wrapped.split("\n"):
            print(f"|      {line}")
        print(f"|")
        print(f"|   [RULE-BASED] Prediction : {rule_result['predicted_category']}")
        print(f"|      {rule_result['confidence_note']}")

        if zs_result:
            print(f"|   [ZERO-SHOT] Prediction  : {zs_result['predicted_category']} "
                  f"({zs_result['confidence']:.2%})")
            print(f"|      Scores: {zs_result['all_scores']}")

        # Match check
        rule_match = "[MATCH]" if rule_result["predicted_category"] == item["expected_category"] else "[MISS]"
        print(f"|")
        print(f"|   Rule-Based Match: {rule_match}")

        if zs_result:
            zs_match = "[MATCH]" if zs_result["predicted_category"] == item["expected_category"] else "[MISS]"
            print(f"|   Zero-Shot Match : {zs_match}")

        print(f"+{'-' * 79}")
        print()

    # ── Summary ──
    rule_correct = sum(
        1 for item in classified_data
        if classify_rule_based(item["transcript"])["predicted_category"]
           == item["expected_category"]
    )

    zs_correct = sum(
        1 for item in classified_data
        if item.get("zero_shot")
        and item["zero_shot"]["predicted_category"] == item["expected_category"]
    )
    zs_total = sum(1 for item in classified_data if item.get("zero_shot"))

    total = len(classified_data)

    print(separator)
    print("  ACCURACY SUMMARY")
    print(separator)
    print(f"  Rule-Based Classifier : {rule_correct}/{total} correct "
          f"({rule_correct/total:.0%})")
    if zs_total:
        print(f"  Zero-Shot Classifier  : {zs_correct}/{zs_total} correct "
              f"({zs_correct/zs_total:.0%})")
    print(separator)
    print()


def save_results_json(classified_data: list[dict]):
    """Save full results to a JSON file."""
    output = []
    for item in classified_data:
        rule_result = classify_rule_based(item["transcript"])
        entry = {
            "call_id": item["call_id"],
            "filename": item["filename"],
            "expected_category": item["expected_category"],
            "transcript": item["transcript"],
            "source": item["source"],
            "rule_based_prediction": rule_result["predicted_category"],
            "rule_based_evidence": rule_result["evidence"],
        }
        if item.get("zero_shot"):
            entry["zero_shot_prediction"] = item["zero_shot"]["predicted_category"]
            entry["zero_shot_confidence"] = item["zero_shot"]["confidence"]
            entry["zero_shot_all_scores"] = item["zero_shot"]["all_scores"]
        output.append(entry)

    out_path = OUTPUT_DIR / "classification_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Results saved to {out_path}")

def main():
    print()
    print("=" * 80)
    print("  SMARTNODE -- CALL TRANSCRIPTION & CLASSIFICATION POC")
    print("=" * 80)
    print()
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Calls     : {len(SAMPLE_CALLS)}")
    print(f"  Categories: Closed | Open | Urgent")
    print()
    audio_available = generate_audio_samples()

    transcripts = transcribe_all_calls(use_audio=audio_available)

    print("[CLASSIFY] Running rule-based classification...\n")
    for t in transcripts:
        result = classify_rule_based(t["transcript"])
        print(f"   {t['call_id']}: {result['predicted_category']:8s}  "
              f"(expected: {t['expected_category']})")
    print()

    classified = classify_all_zero_shot(transcripts)

    display_results(classified)

    save_results_json(classified)

    print("\n[DONE] POC pipeline complete!\n")
    
if __name__ == "__main__":
    main()
