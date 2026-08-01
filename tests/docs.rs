//! Documentation-consistency gates for `docs/MODELS.md`.
//!
//! These are **not** physics. They gate the one property of the claims ledger
//! that a reader relies on and that nothing else checks: that the census line
//! at the top of the section describes the table underneath it.
//!
//! The reason this exists: `CONTRIBUTING.md` requires the census to move in the
//! same change as any added, retired or re-statused row, and on 2026-08-01 it
//! had not. The line claimed 118 rows (80 verified, 10 validated, 16 pinned,
//! 12 ungated) against an actual 122 (83 / 10 / 17 / 12). Nothing failed,
//! because prose cannot fail — so the gap was invisible until someone counted.
//! A reader who takes "10 of 118 claims are validated" as the honest summary of
//! this solver deserves that ratio to be arithmetic rather than recollection.
//!
//! What this does not gate: whether each row's *number* still matches the
//! assertion in the gate it cites. That drifted too, and catching it
//! mechanically is harder, because the ledger quotes numbers in prose. The
//! ground truth for any such number is the `assert!` in the named test.

use std::fs;
use std::path::PathBuf;

/// The four statuses every ledger row must carry, in census order.
const STATUSES: [&str; 4] = ["verified", "validated", "pinned", "ungated"];

fn models_md() -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "docs", "MODELS.md"]
        .iter()
        .collect();
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The `## Claims ledger` section, up to the next top-level heading.
fn claims_ledger(md: &str) -> &str {
    let start = md
        .find("\n## Claims ledger")
        .expect("docs/MODELS.md has no `## Claims ledger` section")
        + 1;
    let rest = &md[start..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    &rest[..end]
}

/// Split one markdown table row into trimmed cells, or `None` if the line is
/// not a data row (header, separator, or not a table row at all).
fn data_row(line: &str) -> Option<Vec<&str>> {
    let line = line.trim();
    if !line.starts_with('|') {
        return None;
    }
    let cells: Vec<&str> = line
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();
    if cells.len() < 4 {
        return None;
    }
    // The header row, and the `|---|---|` separator under it.
    if cells[0].eq_ignore_ascii_case("claim")
        || cells[0].chars().all(|c| c == '-' || c == ':' || c == ' ')
    {
        return None;
    }
    Some(cells)
}

/// Normalise a status cell: `**pinned**` → `pinned`, and
/// `verified *(same lineage)*` → `verified`, since the parenthetical is a flag
/// on the claim rather than a fifth status.
fn status_of(cell: &str) -> Option<&'static str> {
    let stripped = cell.replace('*', "");
    let head = stripped
        .split('(')
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    STATUSES.into_iter().find(|s| *s == head)
}

/// Every digit run in a line, in order — enough to read the census without a
/// regex crate.
fn numbers_in(line: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in line.chars() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(cur.parse().expect("digit run parses"));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.parse().expect("digit run parses"));
    }
    out
}

/// **Every ledger row carries exactly one of the four statuses.**
///
/// A typo'd or missing status would silently skew the census below, and the
/// ledger's own preamble is explicit that a claim which cannot be given one of
/// the four is a claim that is not finished.
#[test]
fn every_claims_ledger_row_has_a_recognised_status() {
    let md = models_md();
    let ledger = claims_ledger(&md);
    let mut bad = Vec::new();
    for line in ledger.lines() {
        if let Some(cells) = data_row(line)
            && status_of(cells[2]).is_none()
        {
            bad.push(format!("  {} → status cell {:?}", cells[0], cells[2]));
        }
    }
    assert!(
        bad.is_empty(),
        "ledger rows whose status is not one of {STATUSES:?}:\n{}",
        bad.join("\n")
    );
}

/// **The census line matches the table it introduces.**
///
/// `CONTRIBUTING.md` § "Documentation to keep in sync" requires the census to
/// be updated in the same change as the row it describes. This is that
/// requirement, enforced.
#[test]
fn the_claims_ledger_census_matches_the_rows() {
    let md = models_md();
    let ledger = claims_ledger(&md);

    let mut counts = [0usize; 4];
    let mut total = 0usize;
    for line in ledger.lines() {
        if let Some(cells) = data_row(line) {
            let status =
                status_of(cells[2]).unwrap_or_else(|| panic!("unrecognised status {:?}", cells[2]));
            let idx = STATUSES
                .iter()
                .position(|s| *s == status)
                .expect("status is one of STATUSES");
            counts[idx] += 1;
            total += 1;
        }
    }

    let census = ledger
        .lines()
        .find(|l| l.trim_start().starts_with("**Census of the "))
        .expect("the claims ledger has no `**Census of the …**` line");
    let stated = numbers_in(census);
    assert_eq!(
        stated.len(),
        5,
        "census line should state the total then the four statuses in the order \
         {STATUSES:?}; found {} numbers in {census:?}",
        stated.len()
    );

    let want = format!(
        "**Census of the {total} rows below: {} verified, {} validated, \
         {} pinned, {} ungated.**",
        counts[0], counts[1], counts[2], counts[3]
    );
    assert_eq!(
        (stated[0], stated[1], stated[2], stated[3], stated[4]),
        (total, counts[0], counts[1], counts[2], counts[3]),
        "the census line has drifted from the table.\n  states: {census}\n  \
         actual: {want}\nUpdate the census in the same change as the row, per \
         CONTRIBUTING.md."
    );
}
