#!/usr/bin/env python
"""Patch the OMSim WavePID study for the DeepCore production (run once on the Mac, then sync + rebuild):

  python cluster/patch_omsim_deepcore.py ~/IceCube/OMSim/simulations/wavepid        # apply (keeps *.orig4 backups)
  python cluster/patch_omsim_deepcore.py ~/IceCube/OMSim/simulations/wavepid --check  # only report what would change

1. src/OMSimWavePIDDetector.cc   --dom_layout file:<path>: one module per line "x y z [qe ...]" in metres, '#' comments
                                 (cluster/deepcore_layout.csv). The qe column is reported, not applied per module:
                                 all 400 DeepCore modules are HQE, so the global --qe_scale is exact.
2. OMSim_WavePID_study.cc        new option --qe_scale F (default 1): multiply the PMT detection probability QE x CE
                                 by F; 1.35 = R7081-02MOD (DeepCore) relative to the standard R7081-02 of the DOM model.
3. include/OMSimSensitiveDetector.hh, src/OMSimSensitiveDetector.cc
                                 m_qeScale read from the args table; handlePMT: p = QE x CE x qe_scale / photon_fraction,
                                 the existing clamp-and-warn now also fires when qe_scale pushes p above 1
                                 (1.35 x 0.245 = 0.33 -> --photon_fraction must be >= 0.34, not 0.25).

Every edit is anchored on an exact line of the revision pasted on 2026-10-16; a missing anchor stops the script
without writing anything, an already-applied edit is skipped (safe to run twice).
"""
import argparse
import re
import sys
from pathlib import Path

FILE_BRANCH = '''} else if (f.size() == 2 && f[0] == "file") {               // file:<path> -> one module per line "x y z [qe]" [m], '#' comments
    std::ifstream layoutFile(f[1]);
    if (!layoutFile) {
        log_critical("--dom_layout file: cannot open '{}'", f[1]);
        throw std::runtime_error("bad --dom_layout file");
    }
    std::string line; double qeLo = 1e9, qeHi = -1e9;
    while (std::getline(layoutFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        double x, y, z, qe = 1.0;
        if (!(ls >> x >> y >> z)) continue;
        ls >> qe;
        layout.emplace_back(x, y, z);
        if (qe < qeLo) qeLo = qe;
        if (qe > qeHi) qeHi = qe;
    }
    if (layout.empty()) {
        log_critical("--dom_layout file: no modules read from '{}'", f[1]);
        throw std::runtime_error("empty --dom_layout file");
    }
    log_info("--dom_layout file: {} modules from {} (qe column {:.2f}..{:.2f}, not applied per module: use --qe_scale)",
             layout.size(), f[1], qeLo, qeHi);
'''

QE_OPTION = '''("qe_scale", po::value<G4double>()->default_value(1.0),
    "Multiply the PMT detection probability (QE x CE) by this factor: 1.35 = DeepCore high-QE R7081-02MOD relative to "
    "the standard R7081-02 of the DOM model. Applied before the --efficiency_cut; with --photon_fraction F the exactness "
    "condition becomes QE x CE x qe_scale <= F (1.35: F >= 0.34)")
'''

PMT_NEW = '''    if (m_qeScale != 1.0)                                          // --qe_scale: HQE PMT relative to the modelled DOM
        info.PMTResponse.detectionProbability *= m_qeScale;
    if (m_photonFraction < 1.0)                                    // undo the --photon_fraction thinning applied at creation
        info.PMTResponse.detectionProbability /= m_photonFraction;
    if (info.PMTResponse.detectionProbability > 1.0) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            log_warning("QE x CE x qe_scale / photon_fraction > 1 for some photons: increase --photon_fraction or lower "
                        "--qe_scale (clamped to 1, hits are biased low)");
        info.PMTResponse.detectionProbability = 1.0;
    }
'''


def indent_block(block, indent):
    return "".join((indent + l if l.strip() else l) for l in block.splitlines(keepends=True))


def insert_before_anchor(text, anchor, block, done_marker):
    """Insert `block` (indented like the anchor line) in front of the line containing `anchor`."""
    if done_marker in text:
        return text, "already applied"
    m = re.search(r"^([ \t]*)" + re.escape(anchor), text, re.M)
    if not m:
        raise SystemExit(f"anchor not found: {anchor!r}")
    return text[: m.start()] + indent_block(block, m.group(1)) + text[m.start():], "applied"


def append_after_anchor(text, anchor, new_line, done_marker):
    if done_marker in text:
        return text, "already applied"
    m = re.search(r"^([ \t]*)" + re.escape(anchor) + r"[^\n]*\n", text, re.M)
    if not m:
        raise SystemExit(f"anchor not found: {anchor!r}")
    return text[: m.end()] + m.group(1) + new_line + "\n" + text[m.end():], "applied"


def patch_detector(text):
    text, s1 = insert_before_anchor(
        text, '} else if ((f.size() == 2 || f.size() == 3) && f[0] == "fill") {', FILE_BRANCH, 'f[0] == "file"')
    old = 'use string:N:S, grid:NX:NY:NZ:S or fill:S[:R])'
    s2 = "n/a"
    if old in text:
        text = text.replace(old, 'use string:N:S, grid:NX:NY:NZ:S, fill:S[:R] or file:<path>)'); s2 = "applied"
    elif "file:<path>)" in text:
        s2 = "already applied"
    if "#include <fstream>" not in text:              # std::ofstream is already used, but be explicit
        text = re.sub(r"^(#include [^\n]+\n)", r"\1#include <fstream>\n#include <sstream>\n", text, count=1, flags=re.M)
    return text, f"file: branch {s1}; error message {s2}"


def patch_study(text):
    text, s = insert_before_anchor(text, '("charged_tracks", po::value<G4double>()->default_value(1.0),', QE_OPTION, '"qe_scale"')
    return text, f"--qe_scale option {s}"


def patch_header(text):
    text, s = append_after_anchor(text, "double m_photonFraction = 1.0;",
                                  "double m_qeScale = 1.0;          ///< --qe_scale: PMT detection probability multiplier (HQE PMTs)",
                                  "m_qeScale")
    return text, f"m_qeScale member {s}"


def patch_sd(text):
    text, s1 = append_after_anchor(
        text, 'try { m_photonFraction = OMSimCommandArgsTable::getInstance().get<G4double>("photon_fraction"); } catch (...) {}',
        'try { m_qeScale = OMSimCommandArgsTable::getInstance().get<G4double>("qe_scale"); } catch (...) {}',
        'get<G4double>("qe_scale")')
    if "detectionProbability *= m_qeScale" in text:
        return text, f"constructor {s1}; handlePMT already applied"
    m = re.search(r"^[ \t]*if \(m_photonFraction < 1\.0\) \{[^\n]*\n.*?^[ \t]*\}\n[ \t]*\}\n", text, re.M | re.S)
    if not m:
        raise SystemExit("anchor not found: the 'if (m_photonFraction < 1.0) {' block in handlePMT")
    text = text[: m.start()] + PMT_NEW + text[m.end():]
    return text, f"constructor {s1}; handlePMT applied"


EDITS = [
    ("src/OMSimWavePIDDetector.cc", patch_detector),
    ("OMSim_WavePID_study.cc", patch_study),
    ("include/OMSimSensitiveDetector.hh", patch_header),
    ("src/OMSimSensitiveDetector.cc", patch_sd),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study_dir", type=Path, help="~/IceCube/OMSim/simulations/wavepid")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    results = []
    for rel, fn in EDITS:                              # all anchors are checked before anything is written
        p = a.study_dir / rel
        if not p.is_file():
            sys.exit(f"missing file: {p}")
        new, status = fn(p.read_text())
        results.append((p, new, status))
    for p, new, status in results:
        changed = new != p.read_text()
        if changed and not a.check:
            bak = p.with_suffix(p.suffix + ".orig4")
            if not bak.exists():
                bak.write_text(p.read_text())
            p.write_text(new)
        print(f"{p.name}: {status}" + ("" if changed else " (no change)") + (" [check only]" if a.check and changed else ""))
    if not a.check:
        print("now: cd ~/IceCube/WavePID && ./cluster/sync_to_fasrc.sh ; on FASRC: sbatch rebuild_fasrc.sbatch")
