import argparse
import re
import textwrap
from pathlib import Path

import requests

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Preformatted,
    KeepTogether,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(value):
    if value is None:
        return ""

    value = str(value)

    # Quelques nettoyages utiles pour le PDF
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    return value.strip()


def escape_xml(text):
    """
    ReportLab Paragraph utilise un sous-ensemble de HTML/XML.
    """
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def safe_filename(name):
    name = re.sub(r"[^\w.-]+", "_", name)
    return name.strip("_")


def parse_requires_dist(requires_dist):
    """
    Transforme les métadonnées PEP 508 en lignes lisibles.
    """
    if not requires_dist:
        return []

    result = []

    for dependency in requires_dist:
        dependency = dependency.strip()

        # Séparation approximative de la partie package/version
        # et de la partie environnement/extras.
        result.append(dependency)

    return result


# ---------------------------------------------------------------------------
# README -> éléments PDF
# ---------------------------------------------------------------------------

def add_readme_content(story, readme, styles):
    """
    Convertit grossièrement un README Markdown en éléments ReportLab.

    L'objectif n'est PAS d'avoir un rendu Markdown parfait.
    L'objectif est d'avoir une structure textuelle excellente pour PageIndex.
    """

    if not readme:
        story.append(
            Paragraph(
                "No README description is available for this package.",
                styles["BodyCustom"],
            )
        )
        return

    lines = readme.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    in_code_block = False
    code_buffer = []

    def flush_code():
        nonlocal code_buffer

        if code_buffer:
            code = "\n".join(code_buffer)

            story.append(
                Preformatted(
                    code,
                    styles["Code"],
                )
            )

            story.append(Spacer(1, 4 * mm))

            code_buffer = []

    for line in lines:
        stripped = line.strip()

        # ---------------------------------------------------------------
        # Code block
        # ---------------------------------------------------------------

        if stripped.startswith("```") or stripped.startswith("~~~"):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True

            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        # ---------------------------------------------------------------
        # Markdown headings
        # ---------------------------------------------------------------

        match = re.match(r"^(#{1,6})\s+(.*)$", stripped)

        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # On limite à trois niveaux pour éviter un arbre trop fragmenté
            if level == 1:
                style = styles["READMEH1"]
            elif level == 2:
                style = styles["READMEH2"]
            else:
                style = styles["READMEH3"]

            story.append(
                Paragraph(
                    escape_xml(title),
                    style,
                )
            )

            story.append(Spacer(1, 2 * mm))
            continue

        # ---------------------------------------------------------------
        # Empty line
        # ---------------------------------------------------------------

        if not stripped:
            story.append(Spacer(1, 2 * mm))
            continue

        # ---------------------------------------------------------------
        # Bullet list
        # ---------------------------------------------------------------

        if re.match(r"^[-*+]\s+", stripped):
            text = re.sub(r"^[-*+]\s+", "", stripped)

            story.append(
                Paragraph(
                    "• " + escape_xml(text),
                    styles["BodyCustom"],
                )
            )

            continue

        # ---------------------------------------------------------------
        # Numbered list
        # ---------------------------------------------------------------

        match = re.match(r"^(\d+)\.\s+(.*)$", stripped)

        if match:
            number = match.group(1)
            text = match.group(2)

            story.append(
                Paragraph(
                    f"{number}. {escape_xml(text)}",
                    styles["BodyCustom"],
                )
            )

            continue

        # ---------------------------------------------------------------
        # Normal paragraph
        # ---------------------------------------------------------------

        story.append(
            Paragraph(
                escape_xml(line),
                styles["BodyCustom"],
            )
        )

    if in_code_block:
        flush_code()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def download_pdf(data: dict, output_path: Path):
    info = data.get("info", {})

    package_name = info.get("name") or "Unknown package"
    version = info.get("version") or "Unknown version"

    summary = clean_text(info.get("summary"))
    description = clean_text(info.get("description"))
    requires_python = clean_text(info.get("requires_python"))

    requires_dist = parse_requires_dist(
        info.get("requires_dist")
    )

    project_urls = info.get("project_urls") or {}

    home_page = info.get("home_page")
    package_page = f"https://pypi.org/project/{package_name}/"

    # -----------------------------------------------------------------------
    # Styles
    # -----------------------------------------------------------------------

    base = getSampleStyleSheet()

    styles = {}

    styles["TitleCustom"] = ParagraphStyle(
        "TitleCustom",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        spaceAfter=8 * mm,
        alignment=TA_LEFT,
    )

    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontSize=12,
        leading=17,
        spaceAfter=8 * mm,
    )

    styles["H1Custom"] = ParagraphStyle(
        "H1Custom",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceBefore=6 * mm,
        spaceAfter=4 * mm,
        keepWithNext=True,
    )

    styles["H2Custom"] = ParagraphStyle(
        "H2Custom",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=4 * mm,
        spaceAfter=3 * mm,
        keepWithNext=True,
    )

    styles["READMEH1"] = ParagraphStyle(
        "READMEH1",
        parent=base["Heading1"],
        fontSize=17,
        leading=21,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        keepWithNext=True,
    )

    styles["READMEH2"] = ParagraphStyle(
        "READMEH2",
        parent=base["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        keepWithNext=True,
    )

    styles["READMEH3"] = ParagraphStyle(
        "READMEH3",
        parent=base["Heading3"],
        fontSize=12,
        leading=15,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
        keepWithNext=True,
    )

    styles["BodyCustom"] = ParagraphStyle(
        "BodyCustom",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=2.5 * mm,
    )

    styles["Small"] = ParagraphStyle(
        "Small",
        parent=base["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.grey,
    )

    styles["Code"] = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=5 * mm,
        rightIndent=5 * mm,
        spaceBefore=2 * mm,
        spaceAfter=3 * mm,
    )

    styles["InstallCode"] = ParagraphStyle(
        "InstallCode",
        fontName="Courier-Bold",
        fontSize=12,
        leading=16,
        leftIndent=8 * mm,
        spaceBefore=3 * mm,
        spaceAfter=5 * mm,
    )

    # -----------------------------------------------------------------------
    # Document
    # -----------------------------------------------------------------------

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"{package_name} - PyPI documentation",
        author="PyPI PageIndex PDF generator",
    )

    story = []

    # -----------------------------------------------------------------------
    # PACKAGE
    # -----------------------------------------------------------------------

    story.append(
        Paragraph(
            escape_xml(package_name),
            styles["TitleCustom"],
        )
    )

    story.append(
        Paragraph(
            f"Python package documentation — version "
            f"<b>{escape_xml(version)}</b>",
            styles["Subtitle"],
        )
    )

    if summary:
        story.append(
            Paragraph(
                escape_xml(summary),
                styles["BodyCustom"],
            )
        )

    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "This document was generated from the official PyPI JSON metadata.",
            styles["Small"],
        )
    )

    # -----------------------------------------------------------------------
    # INSTALLATION
    # -----------------------------------------------------------------------

    story.append(
        Paragraph(
            "1. Installation",
            styles["H1Custom"],
        )
    )

    story.append(
        Paragraph(
            f"To install the Python package <b>{escape_xml(package_name)}</b>, "
            "use pip with the following command:",
            styles["BodyCustom"],
        )
    )

    story.append(
        Preformatted(
            f"pip install {package_name}",
            styles["InstallCode"],
        )
    )

    story.append(
        Paragraph(
            "The command above installs the latest version available from PyPI.",
            styles["BodyCustom"],
        )
    )

    story.append(
        Paragraph(
            f"To install exactly version {escape_xml(version)}, use:",
            styles["BodyCustom"],
        )
    )

    story.append(
        Preformatted(
            f"pip install {package_name}=={version}",
            styles["InstallCode"],
        )
    )

    story.append(
        Paragraph(
            "Alternative installation command using the Python module form of pip:",
            styles["BodyCustom"],
        )
    )

    story.append(
        Preformatted(
            f"python -m pip install {package_name}",
            styles["InstallCode"],
        )
    )

    # -----------------------------------------------------------------------
    # PYTHON REQUIREMENTS
    # -----------------------------------------------------------------------

    story.append(
        Paragraph(
            "2. Python Requirements",
            styles["H1Custom"],
        )
    )

    if requires_python:
        story.append(
            Paragraph(
                f"This package requires Python version: "
                f"<b>{escape_xml(requires_python)}</b>",
                styles["BodyCustom"],
            )
        )
    else:
        story.append(
            Paragraph(
                "PyPI does not specify a Python version requirement "
                "for this release.",
                styles["BodyCustom"],
            )
        )

    # -----------------------------------------------------------------------
    # DEPENDENCIES
    # -----------------------------------------------------------------------

    story.append(
        Paragraph(
            "3. Dependencies",
            styles["H1Custom"],
        )
    )

    if requires_dist:
        story.append(
            Paragraph(
                "The package declares the following dependencies:",
                styles["BodyCustom"],
            )
        )

        for dependency in requires_dist:
            story.append(
                Paragraph(
                    "• " + escape_xml(dependency),
                    styles["BodyCustom"],
                )
            )
    else:
        story.append(
            Paragraph(
                "No runtime dependencies are declared in the PyPI metadata.",
                styles["BodyCustom"],
            )
        )

    # -----------------------------------------------------------------------
    # DESCRIPTION
    # -----------------------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "4. Description and README",
            styles["H1Custom"],
        )
    )

    add_readme_content(
        story,
        description,
        styles,
    )

    # -----------------------------------------------------------------------
    # PROJECT URLS
    # -----------------------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "5. Project URLs",
            styles["H1Custom"],
        )
    )

    if home_page:
        story.append(
            Paragraph(
                f"<b>Homepage:</b> {escape_xml(home_page)}",
                styles["BodyCustom"],
            )
        )

    if project_urls:
        for label, url in project_urls.items():
            story.append(
                Paragraph(
                    f"<b>{escape_xml(label)}:</b> "
                    f"{escape_xml(url)}",
                    styles["BodyCustom"],
                )
            )

    story.append(
        Paragraph(
            f"<b>PyPI project page:</b> "
            f"{escape_xml(package_page)}",
            styles["BodyCustom"],
        )
    )

    # -----------------------------------------------------------------------
    # RELEASE INFORMATION
    # -----------------------------------------------------------------------

    story.append(
        Paragraph(
            "6. Release Information",
            styles["H1Custom"],
        )
    )

    releases = data.get("releases", {})

    # On évite de mettre absolument tous les fichiers de toutes les releases
    # dans le PDF : cela peut devenir énorme.
    release_versions = list(releases.keys())

    story.append(
        Paragraph(
            f"PyPI contains {len(release_versions)} release versions "
            f"for this project.",
            styles["BodyCustom"],
        )
    )

    if release_versions:
        story.append(
            Paragraph(
                "Available release versions:",
                styles["H2Custom"],
            )
        )

        for release in release_versions:
            story.append(
                Paragraph(
                    f"• {escape_xml(release)}",
                    styles["BodyCustom"],
                )
            )

    # -----------------------------------------------------------------------
    # CURRENT RELEASE FILES
    # -----------------------------------------------------------------------

    story.append(
        Paragraph(
            "7. Current Release Files",
            styles["H1Custom"],
        )
    )

    current_files = data.get("urls", [])

    if current_files:
        for file_info in current_files:
            filename = file_info.get("filename", "")
            python_version = file_info.get("python_version", "")
            packagetype = file_info.get("packagetype", "")
            requires = file_info.get("requires_python")

            text = (
                f"<b>{escape_xml(filename)}</b>"
                f" — type: {escape_xml(packagetype)}"
            )

            if python_version:
                text += (
                    f" — Python: {escape_xml(python_version)}"
                )

            if requires:
                text += (
                    f" — requires Python: "
                    f"{escape_xml(requires)}"
                )

            story.append(
                Paragraph(
                    text,
                    styles["BodyCustom"],
                )
            )
    else:
        story.append(
            Paragraph(
                "No release files are available in the API response.",
                styles["BodyCustom"],
            )
        )

    # -----------------------------------------------------------------------
    # SOURCE / METADATA
    # -----------------------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "8. PyPI Metadata",
            styles["H1Custom"],
        )
    )

    metadata_fields = [
        ("Name", info.get("name")),
        ("Version", info.get("version")),
        ("Summary", info.get("summary")),
        ("Author", info.get("author")),
        ("Author email", info.get("author_email")),
        ("Maintainer", info.get("maintainer")),
        ("License", info.get("license")),
        ("Platform", info.get("platform")),
        ("Keywords", info.get("keywords")),
        ("Classifier count", len(info.get("classifiers") or [])),
    ]

    for label, value in metadata_fields:
        if value not in (None, "", []):
            story.append(
                Paragraph(
                    f"<b>{escape_xml(label)}:</b> "
                    f"{escape_xml(value)}",
                    styles["BodyCustom"],
                )
            )

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------

    def add_page_number(canvas, document):
        canvas.saveState()

        canvas.setFont("Helvetica", 7)

        canvas.drawString(
            18 * mm,
            10 * mm,
            f"{package_name} — PyPI / PageIndex",
        )

        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )