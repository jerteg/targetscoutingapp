import { useState, useEffect, useRef, useCallback } from "react";

// ─── Data Constants ───

const FORMATIONS = {
  "4-3-3": {
    lines: [
      { name: "Aanval", positions: ["LW", "ST", "RW"] },
      { name: "Middenveld", positions: ["LCM", "DM", "RCM"] },
      { name: "Verdediging", positions: ["LB", "LCB", "RCB", "RB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "4-4-2": {
    lines: [
      { name: "Aanval", positions: ["LST", "RST"] },
      { name: "Middenveld", positions: ["LM", "LCM", "RCM", "RM"] },
      { name: "Verdediging", positions: ["LB", "LCB", "RCB", "RB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "3-5-2": {
    lines: [
      { name: "Aanval", positions: ["LST", "RST"] },
      { name: "Middenveld", positions: ["LM", "LCM", "AM", "RCM", "RM"] },
      { name: "Verdediging", positions: ["LCB", "CB", "RCB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "4-2-3-1": {
    lines: [
      { name: "Aanval", positions: ["ST"] },
      { name: "Middenveld", positions: ["LW", "AM", "RW"] },
      { name: "Middenveld", positions: ["LDM", "RDM"] },
      { name: "Verdediging", positions: ["LB", "LCB", "RCB", "RB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "3-4-3": {
    lines: [
      { name: "Aanval", positions: ["LW", "ST", "RW"] },
      { name: "Middenveld", positions: ["LM", "LCM", "RCM", "RM"] },
      { name: "Verdediging", positions: ["LCB", "CB", "RCB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "4-1-4-1": {
    lines: [
      { name: "Aanval", positions: ["ST"] },
      { name: "Middenveld", positions: ["LM", "LCM", "RCM", "RM"] },
      { name: "Middenveld", positions: ["DM"] },
      { name: "Verdediging", positions: ["LB", "LCB", "RCB", "RB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "5-3-2": {
    lines: [
      { name: "Aanval", positions: ["LST", "RST"] },
      { name: "Middenveld", positions: ["LM", "CM", "RM"] },
      { name: "Verdediging", positions: ["LWB", "LCB", "CB", "RCB", "RWB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
  "4-3-1-2": {
    lines: [
      { name: "Aanval", positions: ["LST", "RST"] },
      { name: "Middenveld", positions: ["AM"] },
      { name: "Middenveld", positions: ["LCM", "CM", "RCM"] },
      { name: "Verdediging", positions: ["LB", "LCB", "RCB", "RB"] },
      { name: "Keeper", positions: ["GK"] },
    ],
  },
};

const POSITION_TYPES = ["1e optie", "Back-up", "Jeugd"];
const TYPE_COLORS = { "1e optie": "#6366f1", "Back-up": "#22c55e", Jeugd: "#f59e0b" };
const CURRENT_YEAR = 2026;

const FLAG_MAP = {
  NL: "\u{1F1F3}\u{1F1F1}", BEL: "\u{1F1E7}\u{1F1EA}", DEN: "\u{1F1E9}\u{1F1F0}", JAP: "\u{1F1EF}\u{1F1F5}",
  ARG: "\u{1F1E6}\u{1F1F7}", BRA: "\u{1F1E7}\u{1F1F7}", IND: "\u{1F1EE}\u{1F1F3}", GER: "\u{1F1E9}\u{1F1EA}",
  FRA: "\u{1F1EB}\u{1F1F7}", ENG: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}",
  ESP: "\u{1F1EA}\u{1F1F8}", POR: "\u{1F1F5}\u{1F1F9}", ITA: "\u{1F1EE}\u{1F1F9}", CRO: "\u{1F1ED}\u{1F1F7}",
  SUI: "\u{1F1E8}\u{1F1ED}", AUT: "\u{1F1E6}\u{1F1F9}", SWE: "\u{1F1F8}\u{1F1EA}", NOR: "\u{1F1F3}\u{1F1F4}",
  USA: "\u{1F1FA}\u{1F1F8}", MEX: "\u{1F1F2}\u{1F1FD}", COL: "\u{1F1E8}\u{1F1F4}", URU: "\u{1F1FA}\u{1F1FE}",
  CHI: "\u{1F1E8}\u{1F1F1}", PAR: "\u{1F1F5}\u{1F1FE}", SRB: "\u{1F1F7}\u{1F1F8}", POL: "\u{1F1F5}\u{1F1F1}",
  CZE: "\u{1F1E8}\u{1F1FF}", TUR: "\u{1F1F9}\u{1F1F7}", GHA: "\u{1F1EC}\u{1F1ED}", NGA: "\u{1F1F3}\u{1F1EC}",
  CMR: "\u{1F1E8}\u{1F1F2}", SEN: "\u{1F1F8}\u{1F1F3}", MAR: "\u{1F1F2}\u{1F1E6}", EGY: "\u{1F1EA}\u{1F1EC}",
  KOR: "\u{1F1F0}\u{1F1F7}", AUS: "\u{1F1E6}\u{1F1FA}",
  SCO: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F}",
  WAL: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0077}\u{E006C}\u{E0073}\u{E007F}",
  IRE: "\u{1F1EE}\u{1F1EA}", GRE: "\u{1F1EC}\u{1F1F7}", ROU: "\u{1F1F7}\u{1F1F4}",
  UKR: "\u{1F1FA}\u{1F1E6}", ISR: "\u{1F1EE}\u{1F1F1}", CAN: "\u{1F1E8}\u{1F1E6}",
  PER: "\u{1F1F5}\u{1F1EA}", ECU: "\u{1F1EA}\u{1F1E8}", VEN: "\u{1F1FB}\u{1F1EA}",
  CHN: "\u{1F1E8}\u{1F1F3}", ZAF: "\u{1F1FF}\u{1F1E6}", ALG: "\u{1F1E9}\u{1F1FF}",
  CIV: "\u{1F1E8}\u{1F1EE}", other: "\u{1F3F3}\u{FE0F}",
};
const NATIONALITIES = Object.keys(FLAG_MAP);

const COMP_NATIONALITY = {
  Eredivisie: "NL", "Eerste Divisie": "NL", "Premier League": "ENG", "La Liga": "ESP",
  "Serie A": "ITA", Bundesliga: "GER", "Ligue 1": "FRA", "K League 1": "KOR",
  "K League 2": "KOR", "J1 League": "JAP", MLS: "USA", "Liga MX": "MEX",
  "Superliga": "ARG", "Pro League": "BEL",
};

function generateId() { return Math.random().toString(36).substr(2, 9); }

function getDefaultData() {
  return {
    clubs: [{
      id: generateId(), name: "Ajax", competition: "Eredivisie", formation: "4-3-3",
      positions: {
        ST: { "1e optie": [{ id: generateId(), name: "Kasper Dolberg", dob: 1997, nat: "DEN", contract: 2029 }], "Back-up": [], Jeugd: [] },
        LW: { "1e optie": [{ id: generateId(), name: "Mika Godts", dob: 2005, nat: "BEL", contract: 2029 }], "Back-up": [], Jeugd: [] },
        DM: { "1e optie": [{ id: generateId(), name: "Jorthy Mokio", dob: 2008, nat: "BEL", contract: 2031 }], "Back-up": [{ id: generateId(), name: "Davy Klaassen", dob: 1993, nat: "NL", contract: 2027 }], Jeugd: [{ id: generateId(), name: "Youri Baas", dob: 2003, nat: "NL", contract: 2028 }] },
        LCB: { "1e optie": [{ id: generateId(), name: "Ko Itakura", dob: 1997, nat: "JAP", contract: 2029 }], "Back-up": [], Jeugd: [] },
        RCB: { "1e optie": [], "Back-up": [], Jeugd: [{ id: generateId(), name: "Sean Steur", dob: 2008, nat: "NL", contract: 2028 }] },
        RB: { "1e optie": [], "Back-up": [{ id: generateId(), name: "Maher Carrizo", dob: 2006, nat: "ARG", contract: 2030 }], Jeugd: [] },
        LB: { "1e optie": [{ id: generateId(), name: "Lucas Rosa", dob: 2000, nat: "BRA", contract: 2029 }], "Back-up": [{ id: generateId(), name: "Aaron Bouwman", dob: 2007, nat: "NL", contract: 2030 }], Jeugd: [] },
        GK: { "1e optie": [{ id: generateId(), name: "Maarten Paes", dob: 1998, nat: "IND", contract: 2029 }], "Back-up": [], Jeugd: [] },
      },
      shortlists: {},
      notes: "",
    }],
  };
}

// ─── Storage ───
const STORAGE_KEY = "squad-planner-v2";
async function loadData() { try { const r = await window.storage.get(STORAGE_KEY); return r ? JSON.parse(r.value) : null; } catch { return null; } }
async function saveData(data) { try { await window.storage.set(STORAGE_KEY, JSON.stringify(data)); } catch (e) { console.error("Save failed", e); } }

// ─── Helpers ───
function fl(nat) { return FLAG_MAP[nat] || "\u{1F3F3}\u{FE0F}"; }
function age(dob) { return CURRENT_YEAR - dob; }
function contractColor(c) { const l = c - CURRENT_YEAR; return l <= 1 ? "#ef4444" : l <= 2 ? "#f59e0b" : "rgba(148,163,184,0.6)"; }
function shortName(n) { if (!n) return ""; const p = n.split(" "); return p.length === 1 ? p[0] : p[0][0] + ". " + p.slice(1).join(" "); }

// ─── PlayerLine ───
function PlayerLine({ player, onRemove, showClub, actions }) {
  const expiring = player.contract - CURRENT_YEAR <= 1;
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8, padding: "7px 10px",
      background: "rgba(255,255,255,0.03)", borderRadius: 8,
      border: `1px solid ${expiring ? "rgba(239,68,68,0.35)" : "rgba(255,255,255,0.05)"}`,
    }}>
      <span style={{ fontSize: 15, lineHeight: 1, flexShrink: 0 }}>{fl(player.nat)}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 5 }}>
          <span style={{ fontWeight: 600, fontSize: 12, color: "#e2e8f0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{player.name}</span>
          <span style={{ fontSize: 10, color: "#4b5563", flexShrink: 0 }}>{age(player.dob)}</span>
        </div>
        <div style={{ display: "flex", gap: 6, fontSize: 10, marginTop: 1 }}>
          {showClub && player.club && <span style={{ color: "#64748b" }}>{player.club}</span>}
          <span style={{ color: contractColor(player.contract), fontWeight: expiring ? 600 : 400 }}>{"\u27F6"} {player.contract}</span>
        </div>
      </div>
      {actions}
      {onRemove && (
        <button onClick={e => { e.stopPropagation(); onRemove(); }}
          style={{ background: "none", border: "none", color: "rgba(148,163,184,0.3)", cursor: "pointer", fontSize: 14, padding: "0 2px", lineHeight: 1, flexShrink: 0 }}>{"\u00D7"}</button>
      )}
    </div>
  );
}

// ─── Add Player Form ───
function AddPlayerForm({ onAdd, onCancel, showClub }) {
  const [name, setName] = useState("");
  const [dob, setDob] = useState("");
  const [nat, setNat] = useState("NL");
  const [contract, setContract] = useState("");
  const [club, setClub] = useState("");
  const ref = useRef(null);
  useEffect(() => { ref.current?.focus(); }, []);

  const s = { background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "8px 10px", color: "#e2e8f0", fontSize: 12, outline: "none", width: "100%", fontFamily: "inherit" };
  const ok = name && dob && contract;

  return (
    <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: 10, padding: 12, display: "flex", flexDirection: "column", gap: 7, border: "1px solid rgba(255,255,255,0.05)" }}>
      <input ref={ref} placeholder="Naam speler" value={name} onChange={e => setName(e.target.value)} style={s} />
      <div style={{ display: "flex", gap: 6 }}>
        <input placeholder="Geb.jaar" value={dob} onChange={e => setDob(e.target.value)} style={{ ...s, width: showClub ? "22%" : "30%" }} type="number" />
        <select value={nat} onChange={e => setNat(e.target.value)} style={{ ...s, width: showClub ? "22%" : "35%" }}>
          {NATIONALITIES.map(n => <option key={n} value={n}>{fl(n)} {n}</option>)}
        </select>
        <input placeholder="Contract" value={contract} onChange={e => setContract(e.target.value)} style={{ ...s, width: showClub ? "22%" : "35%" }} type="number" />
        {showClub && <input placeholder="Club" value={club} onChange={e => setClub(e.target.value)} style={{ ...s, width: "34%" }} />}
      </div>
      <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
        <button onClick={onCancel} style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, padding: "6px 12px", color: "#64748b", fontSize: 11, cursor: "pointer", fontFamily: "inherit" }}>Annuleren</button>
        <button onClick={() => { if (ok) onAdd({ id: generateId(), name, dob: parseInt(dob), nat, contract: parseInt(contract), ...(showClub && club ? { club } : {}) }); }}
          style={{ background: ok ? "#22c55e" : "rgba(255,255,255,0.03)", border: "none", borderRadius: 6, padding: "6px 12px", color: ok ? "#fff" : "#374151", fontSize: 11, fontWeight: 600, cursor: ok ? "pointer" : "default", fontFamily: "inherit" }}>Toevoegen</button>
      </div>
    </div>
  );
}

// ─── Position Edit Modal with Shortlist ───
function PositionEditModal({ pos, posData, shortlist, onUpdate, onUpdateShortlist, onAddToSquad, onClose }) {
  const [activeType, setActiveType] = useState("1e optie");
  const [showAdd, setShowAdd] = useState(false);
  const [showAddSL, setShowAddSL] = useState(false);

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(6px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        width: "95%", maxWidth: 820, display: "flex",
        background: "linear-gradient(135deg, #1a1f35, #0f1322)",
        border: "1px solid rgba(99,102,241,0.2)", borderRadius: 18,
        boxShadow: "0 30px 80px rgba(0,0,0,0.6)", maxHeight: "85vh", overflow: "hidden",
      }}>
        {/* Left: Squad */}
        <div style={{ flex: 1, padding: 22, overflowY: "auto", borderRight: "1px solid rgba(255,255,255,0.05)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <span style={{ fontWeight: 800, fontSize: 20, color: "#e2e8f0" }}>{pos}</span>
            <button onClick={onClose} style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.06)", color: "#64748b", cursor: "pointer", fontSize: 14, borderRadius: 8, width: 32, height: 32, display: "flex", alignItems: "center", justifyContent: "center" }}>{"\u00D7"}</button>
          </div>

          <div style={{ display: "flex", gap: 4, marginBottom: 14 }}>
            {POSITION_TYPES.map(t => (
              <button key={t} onClick={() => { setActiveType(t); setShowAdd(false); }}
                style={{ flex: 1, padding: "8px 4px", borderRadius: 8, border: activeType === t ? `2px solid ${TYPE_COLORS[t]}` : "1px solid rgba(255,255,255,0.05)", background: activeType === t ? `${TYPE_COLORS[t]}15` : "transparent", color: activeType === t ? "#e2e8f0" : "#374151", fontSize: 11, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>
                {t}{posData?.[t]?.length > 0 && <span style={{ marginLeft: 4, opacity: 0.5, fontSize: 10 }}>({posData[t].length})</span>}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: 12 }}>
            {(posData?.[activeType] || []).map(p => (
              <PlayerLine key={p.id} player={p} onRemove={() => { const u = { ...posData }; u[activeType] = u[activeType].filter(x => x.id !== p.id); onUpdate(u); }} />
            ))}
            {(posData?.[activeType] || []).length === 0 && !showAdd && (
              <div style={{ textAlign: "center", padding: "20px 0", color: "rgba(148,163,184,0.25)", fontSize: 12, fontStyle: "italic" }}>Geen spelers</div>
            )}
          </div>

          {showAdd ? (
            <AddPlayerForm onAdd={p => { const u = { ...posData }; if (!u[activeType]) u[activeType] = []; u[activeType] = [...u[activeType], p]; onUpdate(u); setShowAdd(false); }} onCancel={() => setShowAdd(false)} />
          ) : (
            <button onClick={() => setShowAdd(true)} style={{ width: "100%", padding: 9, borderRadius: 8, border: "1px dashed rgba(255,255,255,0.1)", background: "transparent", color: "#374151", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>+ Speler toevoegen</button>
          )}
        </div>

        {/* Right: Shortlist */}
        <div style={{ width: 310, padding: 22, overflowY: "auto", background: "rgba(0,0,0,0.12)", flexShrink: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 12, color: "#f59e0b", letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 14, display: "flex", alignItems: "center", gap: 6 }}>
            {"\u2605"} Shortlist <span style={{ fontSize: 10, color: "#92400e", fontWeight: 400 }}>{pos}</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: 12 }}>
            {(shortlist || []).map(p => (
              <PlayerLine key={p.id} player={p} showClub
                onRemove={() => onUpdateShortlist((shortlist || []).filter(x => x.id !== p.id))}
                actions={
                  <button onClick={() => onAddToSquad(p)} title="Naar team"
                    style={{ background: "rgba(34,197,94,0.12)", border: "1px solid rgba(34,197,94,0.25)", borderRadius: 5, color: "#22c55e", fontSize: 9, padding: "3px 7px", cursor: "pointer", fontFamily: "inherit", fontWeight: 700, flexShrink: 0, whiteSpace: "nowrap" }}>
                    {"\u2192"} team
                  </button>
                }
              />
            ))}
            {(!shortlist || shortlist.length === 0) && !showAddSL && (
              <div style={{ textAlign: "center", padding: "20px 0", color: "rgba(148,163,184,0.2)", fontSize: 11, fontStyle: "italic" }}>Nog geen targets</div>
            )}
          </div>

          {showAddSL ? (
            <AddPlayerForm showClub onAdd={p => { onUpdateShortlist([...(shortlist || []), p]); setShowAddSL(false); }} onCancel={() => setShowAddSL(false)} />
          ) : (
            <button onClick={() => setShowAddSL(true)} style={{ width: "100%", padding: 9, borderRadius: 8, border: "1px dashed rgba(245,158,11,0.25)", background: "transparent", color: "#78350f", fontSize: 12, cursor: "pointer", fontFamily: "inherit" }}>+ Target toevoegen</button>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Position Slot on Pitch ───
function PositionSlot({ pos, posData, shortlist, onUpdate, onUpdateShortlist, onAddToSquad, isHighlighted }) {
  const [open, setOpen] = useState(false);
  const first = posData?.["1e optie"]?.[0];
  const backup = posData?.["Back-up"]?.[0];
  const youth = posData?.["Jeugd"]?.[0];
  const hasFirst = !!first;
  const slCount = shortlist?.length || 0;

  return (
    <div style={{ position: "relative" }}>
      <button onClick={() => setOpen(true)} style={{
        width: 118, minHeight: 76, borderRadius: 10, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "flex-start", cursor: "pointer", fontFamily: "inherit",
        padding: "7px 5px", transition: "all 0.2s",
        border: isHighlighted ? "2px solid rgba(239,68,68,0.55)" : hasFirst ? "1px solid rgba(34,197,94,0.3)" : "1px solid rgba(255,255,255,0.1)",
        background: hasFirst ? "rgba(34,197,94,0.06)" : isHighlighted ? "rgba(239,68,68,0.07)" : "rgba(255,255,255,0.035)",
      }}>
        <span style={{ fontWeight: 800, fontSize: 10, letterSpacing: "0.05em", color: hasFirst ? "#86efac" : isHighlighted ? "#fca5a5" : "#4b5563", marginBottom: 3 }}>{pos}</span>

        {first ? (
          <span style={{ fontSize: 10.5, fontWeight: 600, color: "#e2e8f0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%", lineHeight: 1.3 }}>
            {fl(first.nat)} {shortName(first.name)}
          </span>
        ) : <span style={{ fontSize: 9, color: "rgba(148,163,184,0.2)" }}>{"\u2014"}</span>}

        {backup && <span style={{ fontSize: 9, color: "#4ade80", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%", marginTop: 2, opacity: 0.85 }}>
          {fl(backup.nat)} {shortName(backup.name)}
        </span>}

        {youth && <span style={{ fontSize: 9, color: "#fbbf24", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: "100%", marginTop: 1, opacity: 0.8 }}>
          {fl(youth.nat)} {shortName(youth.name)}
        </span>}

        {slCount > 0 && (
          <span style={{ position: "absolute", top: -5, right: -5, background: "#f59e0b", color: "#000", fontSize: 8, fontWeight: 800, borderRadius: "50%", width: 16, height: 16, display: "flex", alignItems: "center", justifyContent: "center" }}>
            {slCount}
          </span>
        )}
      </button>

      {open && <PositionEditModal pos={pos} posData={posData} shortlist={shortlist}
        onUpdate={onUpdate} onUpdateShortlist={onUpdateShortlist} onAddToSquad={onAddToSquad}
        onClose={() => setOpen(false)} />}
    </div>
  );
}

// ─── Pitch ───
function FormationPitch({ club, onUpdatePosition, onUpdateShortlist, onAddToSquad }) {
  const formation = FORMATIONS[club.formation];
  if (!formation) return null;
  const allPos = formation.lines.flatMap(l => l.positions);
  const gaps = allPos.filter(p => !(club.positions?.[p]?.["1e optie"]?.length > 0));

  return (
    <div style={{
      background: "linear-gradient(180deg, #1a472a 0%, #15522a 30%, #1a6030 60%, #1a472a 100%)",
      borderRadius: 14, padding: "32px 20px", position: "relative", overflow: "hidden",
      border: "1px solid rgba(255,255,255,0.05)",
    }}>
      <div style={{ position: "absolute", inset: 0, opacity: 0.09, pointerEvents: "none" }}>
        <div style={{ position: "absolute", top: "50%", left: "8%", right: "8%", height: 1, background: "#fff" }} />
        <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", width: 80, height: 80, borderRadius: "50%", border: "1px solid #fff" }} />
        <div style={{ position: "absolute", bottom: 0, left: "25%", right: "25%", height: "12%", borderTop: "1px solid #fff", borderLeft: "1px solid #fff", borderRight: "1px solid #fff" }} />
        <div style={{ position: "absolute", top: 0, left: "25%", right: "25%", height: "12%", borderBottom: "1px solid #fff", borderLeft: "1px solid #fff", borderRight: "1px solid #fff" }} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 16, position: "relative", zIndex: 1 }}>
        {formation.lines.map((line, i) => (
          <div key={i} style={{ display: "flex", justifyContent: "center", gap: 10 }}>
            {line.positions.map(pos => (
              <PositionSlot key={pos} pos={pos} posData={club.positions?.[pos]} shortlist={club.shortlists?.[pos]}
                isHighlighted={gaps.includes(pos)}
                onUpdate={u => onUpdatePosition(pos, u)}
                onUpdateShortlist={sl => onUpdateShortlist(pos, sl)}
                onAddToSquad={p => onAddToSquad(pos, p)} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Analysis ───
function SquadAnalysis({ club }) {
  const formation = FORMATIONS[club.formation];
  if (!formation) return null;
  const allPos = formation.lines.flatMap(l => l.positions);
  const domesticNat = COMP_NATIONALITY[club.competition];

  let total = 0, filled = 0, gaps = [], expiring = [], allAges = [], backupsNeeded = 0, domesticCount = 0;
  const lineAges = {};

  allPos.forEach(pos => {
    const pd = club.positions?.[pos];
    if (pd?.["1e optie"]?.length > 0) filled++; else gaps.push(pos);
    if (!(pd?.["Back-up"]?.length > 0)) backupsNeeded++;
    const line = formation.lines.find(l => l.positions.includes(pos));
    const ln = line?.name || "Overig";
    POSITION_TYPES.forEach(t => {
      (pd?.[t] || []).forEach(p => {
        total++;
        const a = age(p.dob);
        allAges.push(a);
        if (!lineAges[ln]) lineAges[ln] = [];
        lineAges[ln].push(a);
        if (p.contract - CURRENT_YEAR <= 1) expiring.push({ ...p, pos });
        if (domesticNat && p.nat === domesticNat) domesticCount++;
      });
    });
  });

  const avg = allAges.length ? (allAges.reduce((a, b) => a + b, 0) / allAges.length).toFixed(1) : "\u2014";
  const lineAvg = {};
  Object.entries(lineAges).forEach(([n, ages]) => { lineAvg[n] = ages.length ? (ages.reduce((a, b) => a + b, 0) / ages.length).toFixed(1) : "\u2014"; });
  const totalSL = Object.values(club.shortlists || {}).reduce((a, sl) => a + (sl?.length || 0), 0);

  const Stat = ({ label, value, color }) => (
    <div style={{ flex: "1 1 120px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, padding: "10px 12px" }}>
      <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800, color }}>{value}</div>
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 18 }}>
      <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
        <Stat label="Spelers" value={total} color="#6366f1" />
        <Stat label="Posities" value={`${filled}/${allPos.length}`} color={filled === allPos.length ? "#22c55e" : "#f59e0b"} />
        <Stat label="Back-ups nodig" value={backupsNeeded} color={backupsNeeded > 0 ? "#f59e0b" : "#22c55e"} />
        <Stat label="Aflopend contract" value={expiring.length} color={expiring.length > 0 ? "#ef4444" : "#22c55e"} />
        <Stat label="Shortlist targets" value={totalSL} color={totalSL > 0 ? "#f59e0b" : "#374151"} />
        {domesticNat && <Stat label={`${fl(domesticNat)} Binnenlands`} value={domesticCount} color={domesticCount >= 8 ? "#22c55e" : "#6366f1"} />}
      </div>

      {/* Avg age */}
      <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, padding: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: "#4b5563", textTransform: "uppercase", letterSpacing: "0.07em" }}>Gem. leeftijd</span>
          <span style={{ fontSize: 22, fontWeight: 800, color: "#a5b4fc" }}>{avg}</span>
        </div>
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
          {["Aanval", "Middenveld", "Verdediging", "Keeper"].filter(n => lineAvg[n]).map(n => (
            <div key={n} style={{ flex: "1 1 90px", background: "rgba(255,255,255,0.025)", borderRadius: 8, padding: "8px 10px", textAlign: "center" }}>
              <div style={{ fontSize: 9, color: "#374151", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 3 }}>{n}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>{lineAvg[n]}</div>
            </div>
          ))}
        </div>
      </div>

      {gaps.length > 0 && (
        <div style={{ background: "rgba(239,68,68,0.04)", border: "1px solid rgba(239,68,68,0.12)", borderRadius: 10, padding: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#fca5a5", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>Posities zonder 1e optie</div>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
            {gaps.map(g => <span key={g} style={{ background: "rgba(239,68,68,0.1)", color: "#fca5a5", padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>{g}</span>)}
          </div>
        </div>
      )}

      {expiring.length > 0 && (
        <div style={{ background: "rgba(239,68,68,0.03)", border: "1px solid rgba(239,68,68,0.1)", borderRadius: 10, padding: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#fca5a5", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>Aflopende contracten ({"\u2264"}1 jaar)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {expiring.map(p => (
              <div key={p.id} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#e2e8f0" }}>
                <span>{fl(p.nat)} {p.name} <span style={{ color: "#374151" }}>({p.pos})</span></span>
                <span style={{ color: "#ef4444", fontWeight: 600 }}>{p.contract}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Club Card ───
function ClubCard({ club, isSelected, onSelect, onDelete }) {
  const formation = FORMATIONS[club.formation];
  const allPos = formation ? formation.lines.flatMap(l => l.positions) : [];
  const filled = allPos.filter(p => club.positions?.[p]?.["1e optie"]?.length > 0).length;
  let total = 0;
  allPos.forEach(p => POSITION_TYPES.forEach(t => { total += club.positions?.[p]?.[t]?.length || 0; }));

  return (
    <div onClick={onSelect} style={{
      background: isSelected ? "linear-gradient(135deg, rgba(99,102,241,0.08), rgba(99,102,241,0.02))" : "rgba(255,255,255,0.012)",
      border: isSelected ? "1px solid rgba(99,102,241,0.3)" : "1px solid rgba(255,255,255,0.04)",
      borderRadius: 14, padding: "14px 16px", cursor: "pointer", transition: "all 0.2s", minWidth: 190, flexShrink: 0,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontWeight: 800, fontSize: 15, color: "#e2e8f0" }}>{club.name}</div>
          <div style={{ fontSize: 10, color: "#374151", marginTop: 2 }}>{club.competition && `${club.competition} \u00B7 `}{club.formation}</div>
        </div>
        <button onClick={e => { e.stopPropagation(); if (confirm(`"${club.name}" verwijderen?`)) onDelete(); }}
          style={{ background: "none", border: "none", color: "rgba(148,163,184,0.15)", cursor: "pointer", fontSize: 16, padding: 4 }}>{"\u00D7"}</button>
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
        <div><div style={{ fontSize: 9, color: "#1f2937", textTransform: "uppercase", letterSpacing: "0.07em" }}>Spelers</div><div style={{ fontSize: 16, fontWeight: 700, color: "#6366f1" }}>{total}</div></div>
        <div><div style={{ fontSize: 9, color: "#1f2937", textTransform: "uppercase", letterSpacing: "0.07em" }}>Posities</div><div style={{ fontSize: 16, fontWeight: 700, color: filled === allPos.length ? "#22c55e" : "#f59e0b" }}>{filled}/{allPos.length}</div></div>
      </div>
      <div style={{ marginTop: 8, height: 3, borderRadius: 2, background: "rgba(255,255,255,0.03)", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${allPos.length ? (filled / allPos.length) * 100 : 0}%`, background: filled === allPos.length ? "#22c55e" : "#6366f1", borderRadius: 2, transition: "width 0.5s" }} />
      </div>
    </div>
  );
}

// ─── Add Club Modal ───
function AddClubModal({ onAdd, onClose }) {
  const [name, setName] = useState("");
  const [comp, setComp] = useState("");
  const [fm, setFm] = useState("4-3-3");
  const ref = useRef(null);
  useEffect(() => { ref.current?.focus(); }, []);
  const s = { background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, padding: "11px 13px", color: "#e2e8f0", fontSize: 13, outline: "none", width: "100%", fontFamily: "inherit" };

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000 }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ background: "linear-gradient(135deg, #1a1f35, #0f1322)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 20, padding: 28, width: "90%", maxWidth: 460, boxShadow: "0 40px 100px rgba(0,0,0,0.6)" }}>
        <h2 style={{ margin: "0 0 20px", fontSize: 19, fontWeight: 800, color: "#e2e8f0" }}>Nieuwe club toevoegen</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
          <div>
            <label style={{ fontSize: 10, color: "#374151", textTransform: "uppercase", letterSpacing: "0.07em", display: "block", marginBottom: 5 }}>Naam club *</label>
            <input ref={ref} placeholder="bijv. AZ" value={name} onChange={e => setName(e.target.value)} style={s} />
          </div>
          <div>
            <label style={{ fontSize: 10, color: "#374151", textTransform: "uppercase", letterSpacing: "0.07em", display: "block", marginBottom: 5 }}>Competitie</label>
            <input placeholder="bijv. Eredivisie" value={comp} onChange={e => setComp(e.target.value)} style={s} />
          </div>
          <div>
            <label style={{ fontSize: 10, color: "#374151", textTransform: "uppercase", letterSpacing: "0.07em", display: "block", marginBottom: 5 }}>Formatie</label>
            <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
              {Object.keys(FORMATIONS).map(f => (
                <button key={f} onClick={() => setFm(f)}
                  style={{ padding: "7px 12px", borderRadius: 8, border: fm === f ? "2px solid #6366f1" : "1px solid rgba(255,255,255,0.06)", background: fm === f ? "rgba(99,102,241,0.1)" : "transparent", color: fm === f ? "#a5b4fc" : "#374151", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>{f}</button>
              ))}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 24 }}>
          <button onClick={onClose} style={{ flex: 1, padding: 11, borderRadius: 10, border: "1px solid rgba(255,255,255,0.06)", background: "transparent", color: "#64748b", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>Annuleren</button>
          <button onClick={() => { if (name) onAdd({ id: generateId(), name, competition: comp, formation: fm, positions: {}, shortlists: {}, notes: "" }); }}
            style={{ flex: 2, padding: 11, borderRadius: 10, border: "none", background: name ? "linear-gradient(135deg, #22c55e, #16a34a)" : "rgba(255,255,255,0.03)", color: name ? "#fff" : "#1f2937", fontSize: 13, fontWeight: 700, cursor: name ? "pointer" : "default", fontFamily: "inherit" }}>Club toevoegen</button>
        </div>
      </div>
    </div>
  );
}

// ─── Main ───
export default function SquadPlanner() {
  const [data, setData] = useState(null);
  const [selId, setSelId] = useState(null);
  const [showAdd, setShowAdd] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => { loadData().then(saved => { const d = saved || getDefaultData(); setData(d); if (d.clubs.length) setSelId(d.clubs[0].id); setLoading(false); }); }, []);
  useEffect(() => { if (data) saveData(data); }, [data]);

  const updateClub = useCallback((id, fn) => { setData(prev => ({ ...prev, clubs: prev.clubs.map(c => c.id === id ? fn(c) : c) })); }, []);

  const sel = data?.clubs.find(c => c.id === selId);

  if (loading) return <div style={{ minHeight: "100vh", background: "#0a0e1a", display: "flex", alignItems: "center", justifyContent: "center", color: "#374151", fontFamily: "'DM Sans',sans-serif" }}>Laden...</div>;

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(160deg, #0a0e1a 0%, #0f1629 40%, #0a0e1a 100%)", fontFamily: "'DM Sans',sans-serif", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet" />

      <div style={{ padding: "20px 24px", borderBottom: "1px solid rgba(255,255,255,0.03)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 21, fontWeight: 800, letterSpacing: "-0.03em", background: "linear-gradient(135deg, #e2e8f0, #6366f1)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Squad Planner</h1>
          <p style={{ margin: "2px 0 0", fontSize: 11, color: "#1f2937" }}>Transferwindow analyse</p>
        </div>
        <button onClick={() => setShowAdd(true)} style={{ padding: "9px 16px", borderRadius: 10, border: "none", background: "linear-gradient(135deg, #6366f1, #4f46e5)", color: "#fff", fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: "inherit", display: "flex", alignItems: "center", gap: 5, boxShadow: "0 4px 16px rgba(99,102,241,0.2)" }}>
          <span style={{ fontSize: 16, lineHeight: 1 }}>+</span> Nieuwe club
        </button>
      </div>

      <div style={{ padding: "14px 24px", display: "flex", gap: 9, overflowX: "auto", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
        {data.clubs.map(c => (
          <ClubCard key={c.id} club={c} isSelected={c.id === selId} onSelect={() => setSelId(c.id)}
            onDelete={() => { setData(prev => ({ ...prev, clubs: prev.clubs.filter(x => x.id !== c.id) })); if (selId === c.id) { const r = data.clubs.filter(x => x.id !== c.id); setSelId(r.length ? r[0].id : null); } }} />
        ))}
        {!data.clubs.length && <div style={{ padding: "30px 0", textAlign: "center", width: "100%", color: "#1f2937", fontSize: 13 }}>Voeg een club toe om te beginnen</div>}
      </div>

      {sel && (
        <div style={{ padding: "20px 24px", maxWidth: 920, margin: "0 auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 19, fontWeight: 800 }}>{sel.name}</h2>
              {sel.competition && <span style={{ fontSize: 11, color: "#374151" }}>{sel.competition}</span>}
            </div>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {Object.keys(FORMATIONS).map(f => (
                <button key={f} onClick={() => updateClub(sel.id, c => ({ ...c, formation: f }))}
                  style={{ padding: "5px 10px", borderRadius: 7, border: sel.formation === f ? "2px solid #6366f1" : "1px solid rgba(255,255,255,0.05)", background: sel.formation === f ? "rgba(99,102,241,0.08)" : "transparent", color: sel.formation === f ? "#a5b4fc" : "#1f2937", fontSize: 11, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>{f}</button>
              ))}
            </div>
          </div>

          <FormationPitch club={sel}
            onUpdatePosition={(pos, u) => updateClub(sel.id, c => ({ ...c, positions: { ...c.positions, [pos]: u } }))}
            onUpdateShortlist={(pos, sl) => updateClub(sel.id, c => ({ ...c, shortlists: { ...c.shortlists, [pos]: sl } }))}
            onAddToSquad={(pos, player) => {
              updateClub(sel.id, c => {
                const pd = { "1e optie": [], "Back-up": [], Jeugd: [], ...(c.positions?.[pos] || {}) };
                const target = pd["1e optie"].length === 0 ? "1e optie" : "Back-up";
                const { club: _, ...clean } = player;
                pd[target] = [...pd[target], { ...clean, id: generateId() }];
                const sl = (c.shortlists?.[pos] || []).filter(p => p.id !== player.id);
                return { ...c, positions: { ...c.positions, [pos]: pd }, shortlists: { ...c.shortlists, [pos]: sl } };
              });
            }}
          />

          <div style={{ display: "flex", gap: 12, marginTop: 12, justifyContent: "center", flexWrap: "wrap" }}>
            {[
              { c: "rgba(34,197,94,0.3)", bg: "rgba(34,197,94,0.06)", l: "1e optie ingevuld" },
              { c: "rgba(239,68,68,0.55)", bg: "rgba(239,68,68,0.07)", l: "Geen 1e optie" },
              { c: "#f59e0b", bg: "transparent", l: "\u2605 = shortlist" },
            ].map(x => (
              <div key={x.l} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 11, height: 11, borderRadius: 3, background: x.bg, border: `1px solid ${x.c}` }} />
                <span style={{ fontSize: 10, color: "#374151" }}>{x.l}</span>
              </div>
            ))}
          </div>

          <SquadAnalysis club={sel} />

          <div style={{ marginTop: 16 }}>
            <label style={{ fontSize: 10, color: "#374151", textTransform: "uppercase", letterSpacing: "0.07em", display: "block", marginBottom: 7 }}>Notities &amp; Transferdoelen</label>
            <textarea value={sel.notes || ""} onChange={e => updateClub(sel.id, c => ({ ...c, notes: e.target.value }))} placeholder="Budget, prioriteiten, deadlines..."
              style={{ width: "100%", minHeight: 70, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, padding: 12, color: "#94a3b8", fontSize: 12, fontFamily: "inherit", outline: "none", resize: "vertical", lineHeight: 1.6 }} />
          </div>
        </div>
      )}

      {showAdd && <AddClubModal onAdd={c => { setData(prev => ({ ...prev, clubs: [...prev.clubs, c] })); setSelId(c.id); setShowAdd(false); }} onClose={() => setShowAdd(false)} />}
    </div>
  );
}