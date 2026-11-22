// Dashboard test: 51 participants x 8 reps shown by the page must equal data/processed/rep_features.csv and the saved nested predictions.
// Run: NODE_PATH=$(npm root -g) node tests/dashboard_test.js .
const { chromium } = require("playwright"); const fs = require("fs"); const { spawn } = require("child_process"); const path = require("path");
const root = path.resolve(process.argv[2] || "."); const csv = f => { const [h, ...r] = fs.readFileSync(path.join(root, f), "utf8").trim().split(/\r?\n/); const k = h.split(","); return r.map(l => Object.fromEntries(l.split(",").map((v, i) => [k[i], v]))); };
(async () => {
  const rep = csv("data/processed/rep_features.csv"), pred = csv("results/nested_selection/nested_selection_predictions.csv");
  const b = await chromium.launch(), pg = await b.newPage({ viewport: { width: 1300, height: 1000 } }); const errs = [], fails = []; let checked = 0;
  pg.on("pageerror", e => errs.push(String(e))); pg.on("console", m => m.type() === "error" && errs.push(m.text()));
  await pg.goto("http://localhost:8765/"); await pg.waitForSelector("#person option", { state: "attached" });
  const ids = await pg.$$eval("#person option", o => o.map(x => x.value)); if (ids.length !== 51) fails.push("participants " + ids.length);
  for (const id of ids) {
    await pg.selectOption("#person", id); const rows = rep.filter(r => String(+r.subject) === String(+id)).sort((a, b) => +a.rep - +b.rep);
    if (rows.length !== 8) fails.push(id + " dataset reps " + rows.length);
    const n = await pg.$$eval("#repButtons button", x => x.length); if (n !== 8) fails.push(id + " rep buttons " + n);
    for (let i = 0; i < 8; i++) {
      await pg.click(`#repButtons button:nth-child(${i + 1})`);
      const eff = +(await pg.textContent("#eff")), dur = parseFloat(await pg.textContent("#duration")), end = (await pg.textContent("#endpoint")).replace(/[^0-9.]/g, "");
      const d = rows[i]; if (Math.abs(eff - +d.path_efficiency) > 0.0006) fails.push(`${id} r${i + 1} eff ${eff} vs ${d.path_efficiency}`);
      if (Math.abs(dur - +d.movement_duration_s) > 0.006) fails.push(`${id} r${i + 1} dur`);
      if (d.position_error_mm && end && Math.abs(+end - +d.position_error_mm) > 0.6) fails.push(`${id} r${i + 1} endpoint ${end} vs ${d.position_error_mm}`);
      checked += 3;
    }
    const tbl = await pg.$$eval("#repTable tr", t => t.length); if (tbl !== 9) fails.push(id + " summary table rows " + tbl);
    await pg.click("#reveal"); const txt = await pg.textContent("#answer"); const p = pred.find(r => String(+r.subject) === String(+id));
    const call = +p.predicted_target ? "DCP" : "TD"; if (!txt.includes("Held-out call: " + call)) fails.push(`${id} nested call expected ${call}`); if (!txt.includes(p.winner.replace(/_/g, " "))) fails.push(`${id} winner ${p.winner} not shown`); checked++;
  }
  const plotInk = await pg.evaluate(() => { const c = document.getElementById("plot"), d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 16) if (d[i] > 120) n++; return n; }); if (plotInk < 30) fails.push("plot blank");
  if (!(await pg.textContent(".badge.research"))) fails.push("research badge");
  await pg.screenshot({ path: path.join(root, "tests/dashboard_study.png") });
  await pg.click("#liveTab"); const ex = await pg.textContent(".badge.exploratory"); if (!/not the validated 84-feature/.test(ex)) fails.push("live banner"); await pg.screenshot({ path: path.join(root, "tests/dashboard_live.png") });
  console.log(JSON.stringify({ participants: ids.length, values_checked: checked, page_errors: errs, failures: fails.length, details: fails.slice(0, 10) })); await b.close(); process.exit(fails.length || errs.length ? 1 : 0);
})();
