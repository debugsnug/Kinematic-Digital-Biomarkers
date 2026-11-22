# Reach Lab demonstration (study replay + connected-device session)

The demonstration is the IoT-facing part of the project: a camera or device sends timestamped wrist coordinates to a local receiver, which shows the movement and the same measurements used in the study. It sits on top of the research pipeline and reads its results; it adds no new headline claim.

## Start it

```sh
python3 present.py --open        # or double-click Present.command (macOS); Python 3.9+, standard library only
```
Standalone, no Python: open `presentation/Offline replay.html` in a browser. Slides: `presentation/Reach Lab.pptx` (PDF copy alongside).
The replay data (`demo/study.json`) and the offline page are rebuilt by `python run_pipeline.py` (step 38) or `python demo/build.py`.

## What the numbers on screen mean
| Item | Value | Status |
|---|---|---|
| Headline held-out call per participant | fully nested model selection: **42/51** (accuracy 0.824, 95% CI 0.706-0.922) | selection-unbiased internal estimate |
| Per-fold top-3 majority vote | 44/51 (0.863) | course "validate 2-3 models" requirement |
| Random Forest used for explanations | 47/51 | explanatory only; chosen after seeing all folds, so **optimistic** |
| Wrist-only logistic model (3 features, used for live sessions) | 45/51 | exploratory; features chosen knowing this cohort; **optimistic**; not validated on any new camera |

Earlier decks quoted 48/51 (168 descriptors including joint angles, Random Forest chosen post hoc). That figure is superseded and is not used anywhere in the demo or slides.

## Four-minute rehearsal
| Time | Action |
|---|---|
| 0:00-0:40 | Open the deck. Make a sideways reach. State the task before any accuracy: **DCP = 1 (30 people) vs TD = 0 (21 people), one row per person, eight reaches each.** |
| 0:40-1:40 | **Study replay**, participant **67** (a DCP case, called DCP). Play, scrub, enable **All eight trails**. Path efficiency = straight-line distance / distance travelled. |
| 1:40-2:30 | Reveal 67, then **26** (DCP, called TD: a missed case) and **46** (TD, called DCP: a false alarm). Say: 9 of 51 are wrong in the nested estimate (4 missed DCP, 5 false DCP). |
| 2:30-3:35 | **Live session**: attach a bright marker to the wrist, enable camera, click the marker, **Start reach** / **Finish reach**, eight reaches. The live score is exploratory. |
| 3:35-4:00 | Show capture -> HTTP receiver. Close with limits (below) and the next experiment: new camera vs an independent reference, and an independent, site-balanced cohort. |

## Connected-device mode
On a **trusted local network**: `python3 present.py --host 0.0.0.0 --port 8765`; open `http://localhost:8765`, choose **Live session** -> **Watch device stream**. The receiver has no authentication; the default launcher binds to loopback only.
Sender: `python3 present.py --port 8767 --relay http://RECEIVER_IP:8765 --open` (open the sender dashboard on `localhost` so Chrome allows the camera). Camera-free rehearsal: `python3 demo/send_session.py http://localhost:8765 --stream`, then **Load latest device session**.
Device format: POST JSON to `/stream` with `device`, `rep` (1-8), `path` = up to 1,200 samples `[time_ms, wrist_x_px, wrist_y_px]`; POST completed sessions to `/sessions` with `device` and `reps` (eight paths). Timestamps must increase.

## Limits to state when presenting
Internal subject-wise cross-validation on 51 participants from one dataset; no external test cohort. No site/session/camera variable exists, and recording clip length alone separates the groups (balanced accuracy 0.76), so recording setup cannot be separated from group (`results/confound_analysis`, `results/duration_audit`). A coloured marker, a different camera, frame rate or manual start/finish changes measurements; live scores need validation against a reference system. Study coordinates are 25 Hz, ~2 mm per pixel (source study estimate).

## Attribution
Data adapted from Haberfehlner, Vanmechelen, Van Wonterghem, Aerts, Hallez and Monbaliu, KU Leuven Research Data Repository v2.1, CC BY 4.0, https://doi.org/10.48804/BZM9SJ.
