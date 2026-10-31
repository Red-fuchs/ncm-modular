# Publish checklist (NOTHING HERE HAS BEEN RUN)

Nothing in this package uploads, creates a repo or pushes. Run these yourself, in order, after a last look.

## Pre-flight (local, safe)
```bash
cd ~/Claude-Workspace/ncm-modular
git status --short && git log --format='%h %an <%ae> %s'   # review; author must be neutral
grep -rnE "/Users/|@[a-z0-9-]+\.[a-z]{2,}|hf_[A-Za-z0-9]{10,}|ancap|Obsidian|iCloud" . --exclude-dir=.git --exclude=PUBLISH.md   # must be empty
du -sh . ; git count-objects -vH                           # expect well under 5 MB
```
Optional: drop this file before publishing (`git rm PUBLISH.md && git commit -m "Drop publish notes"`).

## GitHub (public, pushes current branch)
```bash
gh auth status
gh repo create Red-fuchs/ncm-modular --public --source . --push \
  --description "Modular Next-Concept Modeling on Apple Silicon (MLX): codec port, surrogate masking, block-seam experiment, honest logs"
gh repo edit Red-fuchs/ncm-modular --add-topic ncm --add-topic next-concept-modeling --add-topic mlx \
  --add-topic apple-silicon --add-topic text-generation --add-topic diffusion-lm --add-topic vq-vae \
  --add-topic modular-llm --add-topic local-llm --add-topic research-code
```
Suggested description: "Modular Next-Concept Modeling on Apple Silicon (MLX): codec port, surrogate masking, block-seam experiment, honest logs". Topics: ncm, next-concept-modeling, mlx, apple-silicon, text-generation, diffusion-lm, vq-vae, modular-llm, local-llm, research-code.

## Hugging Face (model repo, no weights; README.md front matter is the model card)
```bash
hf auth whoami
hf repo create Red-fuchs/ncm-modular
hf upload Red-fuchs/ncm-modular . . --repo-type model \
  --commit-message "Initial release: MLX codec port, masking, MASSIVE pilot, seam experiment" \
  --exclude ".git/*" "PUBLISH.md" "**/__pycache__/*" "w/*" "*.safetensors" "*.pt" "*.npz"
```
`hf repo create` defaults to a model repo; add `--private` first if you want a look before it goes public. Check the exact flags with `hf repo create --help` and `hf upload --help` for your version.

## Decisions already taken
- Name `ncm-modular`; python folder `src/ncm_modular`. Credit to HazyResearch NCM and janmus1c/ncm-mlx is in README and NOTICE.
- TinyStories gold excerpts in `results/seam/samples.md` kept, with attribution + CDLA-Sharing-1.0 header.
- Organisation names and common fictional-looking names in test sentences kept; real person names replaced; outputs re-run.
- Gemma / JANG converter and benches stay excluded (LICENSES.md, UNCLEAR). Only unverified side-note numbers remain in RESULTS.md section 5.

## Still open
- MASSIVE: confirm CC BY 4.0 on the dataset card you downloaded (only aggregate metrics are shipped).
- Courtesy note to the janmus1c / HazyResearch authors is your call (nothing was sent).
- The TinyStories codec round trip (62 %) has no stored log.
