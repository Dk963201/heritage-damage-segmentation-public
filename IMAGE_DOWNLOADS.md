# Candidate photo downloads

The GitHub release `photos-v0.1-candidates` is the destination for ten ZIP packages of 80 images each. All 800 files passed decoding and archive-content SHA-256 checks. Counts:609 Erbil and 191 Amedi. Ten ZIPs total 2,023,945,223 bytes. Minimum short dimension:1536 pixels. All 800 downloaded SHA-256 hashes are distinct; visual near-duplicates remain unchecked.

These are API-advertised image versions. Most are resized derivatives; where the source was smaller, its original-size version may be returned. Original source hashes are not hashes of derivatives. Each image has a per-file attribution JSON recording its source page, original URL, downloaded URL, photographer, license URL, actual dimensions and downloaded SHA-256.

Download the ZIP assets from Releases and extract all packages into the same directory. Image IDs are unique across packages. Keep `attribution/` with `images/`; preserve author credits and license notices, including ShareAlike conditions where applicable.

The images are unannotated candidates. Download integrity does not prove masonry relevance, absence of near-duplicates, structural condition or suitability for training. No official dataset approval is claimed.

`download_image_versions.py` resumes from its local ledger and stops on HTTP 403/429. It uses Python 3.14+ and Pillow; do not run it to bypass a source restriction. Wait for restrictions to clear and obey any Retry-After instructions. Local images and archives remain excluded from Git history.
