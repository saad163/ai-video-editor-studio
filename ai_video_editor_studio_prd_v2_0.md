# AI Video Editor Studio (PRD v2.0)
**Version:** 2.0 (Optimized MVP Scope)  
**Target Platform:** Windows Desktop Application (EXE)  
**Primary Architecture:** Local Hardware Processing + Low-Cost Cloud API Fallback  
**Commercial Budget Target:** PKR 150,000  

---

## 1. Executive Summary & Key Modifications
Yeh document AI Video Editor Studio ka revised technical framework hai. Original PRD v1.0 se Template 3 (YouTube Downloader/Interview Cutter) ko bandwidth, IP ban, aur copyright issues ki wajah se Phase 1 ke scope se kharij kar diya gaya hai.

Yeh optimized version 2 core creation templates par focus karta hai:
1. **Template 1: Script & Character-Based Video Assembler**
2. **Template 2: Episode Scene Compiler & Context Matcher**

Is architecture ka maqsad client ke liye ongoing operational cost ko **$0.05 – $0.20 per video** tak mehdood rakhna hai jabke developer ke machine aur client ke target PC par stable performance deliver karna hai.

---

## 2. Hardware Architecture & Hardware Adaptation Strategy

### Client Target Hardware
* **CPU:** AMD Ryzen 5 5600
* **RAM:** 16GB (32GB recommended for large batch renders)
* **GPU:** NVIDIA GeForce RTX 2070 (8GB VRAM)
* **Storage:** 238GB SSD (OS & App), 466GB HDD (Scratch & Output directory)

### Developer Low-Spec Adaptation Strategy (Intel Core i5-1145G7, 16GB RAM, Iris Xe)
* **Dual Execution Mode:**
  * **Dev Mode (Local Laptop):** CPU-based lightweight models (`faster-whisper-tiny` / `small`), mock renders, aur low-resolution 30-second proxy segments testing.
  * **Client / Production Mode:** CUDA acceleration enabled (`NVENC` hardware encoding via FFmpeg, `faster-whisper-medium` run on RTX 2070 8GB VRAM).
* **Storage Guard Engine:** Automated cleanup script jo temporary intermediate audio clips, frame extractions, aur PySceneDetect log files ko render ke foran baad wipe out karega taake disk space exhaust na ho.

---

## 3. Cost-Optimization Engine & API Strategy

| Task | Engine / Provider | Execution Location | Running Cost to Client |
| :--- | :--- | :--- | :--- |
| **Voiceover Transcription** | `faster-whisper` (medium-int8) | Local NVIDIA RTX 2070 | **$0.00 / Free** |
| **Word Alignment & Subtitles** | `stable-ts` / Whisper timestamps | Local CPU / GPU | **$0.00 / Free** |
| **Scene Splitting** | `PySceneDetect` (ContentAware) | Local CPU / GPU | **$0.00 / Free** |
| **Visual Embeddings & Matching**| CLIP / Gemini 1.5 Flash API | Cloud API (Key provided by client) | **~$0.05 – $0.15 per 30m video** |
| **Video Composition & Ken Burns** | `FFmpeg` (with `h264_nvenc`) | Local Machine | **$0.00 / Free** |
| **Media Cache & Deduplication** | `SQLite` Database | Local Disk | **$0.00 / Free** |

---

## 4. Module Specifications

### Module A: Template 1 — Script-Based Video Assembler
* **Inputs:** Voiceover Audio (`.mp3`/`.wav`), Script (`.txt`/`.docx`), Character Images Directory (Folders organized by character/entity name), Footage Directory.
* **Process Flow:**
  1. Local Whisper engine audio transcribe karke timestamped word segments banata hai.
  2. Text parsing script se character names identify karke corresponding folder se match karti hai.
  3. Visual engine image transitions ke liye automated dynamic pan/zoom (Ken Burns Effect) FFmpeg matrix generate karta hai.
  4. Video clips aur static images ka configurable balance (e.g., 60% footage, 40% images).
  5. Subtitle rendering engine customizable text, stroke, background boxes, aur word-level sync burn karta hai.
* **Output:** 1080p / 2K 60FPS MP4 video.

### Module B: Template 2 — Episode Scene Compiler
* **Inputs:** 1 Complete Voiceover Audio File (20–35 minutes), Multiple Source Video Files (e.g., 6–8 Episodes), Optional Reference Script.
* **Process Flow:**
  1. `PySceneDetect` poore episodes ko scene-by-scene split karke har scene ka 1 middle keyframe extract karta hai.
  2. Local hash aur timestamp table `scenes.db` mein store hoti hai.
  3. Voiceover narration ko 15–30 second context chunks mein divide kiya jata hai.
  4. Context chunk text aur scene keyframe descriptions ko Gemini 1.5 Flash API ke batch prompt mein bhej kar relevance score match kiya jata hai.
  5. Selected segments ko FFmpeg cut points de kar lossless stream-copy ya fast NVENC re-encode se narration timeline par sync kiya jata hai.
* **Anti-Duplication:** SQLite database tracking ensure karegi ke aik scene aik hi video mein ya pichle projects mein dohrane se pehle user confirmation mangey.

### Module C: Visual Card Grid Editor & Preview
* Complex Premiere-style multitrack timeline ke bajaye application mein **Scene Card Grid** implement hoga:
  * Har card aik detected audio sentence + selected video/image clip represent karega.
  * User single-click se kisi bhi card ka media replace, delete ya timing extend kar sakega.
  * Embedded lightweight video player (PySide6 `QMediaPlayer`) realtime local preview show karega.

---

## 5. Development Phases & Roadmap

### Phase 1: Core Engine & Architecture Setup (Days 1 – 7)
* **Goal:** Zero-cost local transcription and media processing pipeline proof-of-concept.
* **Deliverables:**
  * Virtual Environment configuration with PySide6, FFmpeg wrapper, and `faster-whisper`.
  * Audio transcription engine with local subtitle generation (`.srt` / `.ass`).
  * Basic SQLite database schema setup for media hashing and anti-duplication records.
  * FFmpeg automation script: Ken Burns zoom effect on static images + audio merge.

### Phase 2: Template 1 Implementation (Days 8 – 14)
* **Goal:** End-to-end working Script & Character-based automation pipeline.
* **Deliverables:**
  * Script text and character folder parser.
  * Automated audio-to-image synchronization logic.
  * Subtitle engine with custom styling (font, outline, color, position).
  * 1080p/2K video export module with hardware acceleration fallback.

### Phase 3: Template 2 & Smart Scene Matcher (Days 15 – 22)
* **Goal:** Multi-episode automated scene detection and voiceover synchronization.
* **Deliverables:**
  * `PySceneDetect` scene split integration.
  * Frame sampling and Gemini 1.5 Flash batch-context matching script.
  * Anti-repetition algorithm using local hash verification.
  * Automated assembly of selected episode clips according to narration timestamps.

### Phase 4: UI Development, Review Grid & Packaging (Days 23 – 30)
* **Goal:** Complete desktop interface, testing, and production packaging.
* **Deliverables:**
  * PySide6 modern dashboard matching AutoCut Studio style.
  * Scene Card Grid for manual review and clip replacement.
  * Settings manager (API key insertion, storage paths, resolution selectors).
  * Windows standalone EXE packaging via Nuitka or PyInstaller.
  * Stress testing on client machine (RTX 2070) via remote environment.

---

## 6. Financial Structure & Milestones (PKR 150,000 Scope)

* **Advance Milestone (PKR 45,000 - 30%):** Project initiation, Phase 1 POC completion, and UI skeleton demo.
* **Mid-Way Milestone (PKR 55,000 - 36.6%):** Template 1 fully functional and Template 2 scene-detection operational.
* **Final Delivery Milestone (PKR 50,000 - 33.3%):** Complete EXE build delivery, Client PC installation, and accepted test video render.
* **Client Operating Expenses:** Client will configure their own Gemini API key inside software settings; developer carries zero cloud liability.