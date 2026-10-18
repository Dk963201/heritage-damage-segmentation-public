# Historical public academic report

This is a sanitized historical edition of the earlier 18 October 2026 public-data development report. It preserves its scientific evidence and explicit limitations. It is not the latest project-readiness report; consult docs/PROJECT_SCOPE.md and REVIEWED_IMPROVEMENT_RESULTS.md in the current repository. The reported 73.25% UK development Dice is not the current Erbil target-development result and is not an independent test.

Student and recipient identification, user-identifying affiliation, private course/source references, and author/machine metadata were removed. Public research author names, dataset citations and license attribution were retained. The masonry comparison figure contains no personal identification and has no EXIF metadata.

The original source files and earlier history remain in the private working copy. This public distribution includes only the sanitized report edition.

## Public report files

- ACADEMIC_REPORT_HISTORICAL_PUBLIC.docx
- ACADEMIC_REPORT_HISTORICAL_PUBLIC.pdf
- build_academic_report_public.py
- public-validation-comparison.jpg

To regenerate the DOCX, put the generator and figure together in a fresh directory and run `python build_academic_report_public.py` with python-docx and lxml installed. The generator refuses to replace an existing report and strips personal metadata from its DOCX output. The supplied PDF was rendered from this DOCX and had all author/XMP metadata removed; its rendered pixels were verified unchanged by that final metadata pass.

## Verification scope

All seven rendered pages were visually inspected. Automated checks examined every DOCX XML/relationship entry, the public generator, PDF text/links/metadata, the embedded figure, and original-file hashes. Scientific metric values were checked against the historical source. Known private identifiers, university domains, email addresses and local-user paths were absent. This is a scoped sanitization check, not a universal guarantee that a document contains no indirect identifying context.

The `qa-render` directory is internal visual-QA material; it is not part of the publication file list.
