import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
} from "cursor/canvas";

/**
 * NaCl 18³ — clean W=1,2,4,6,12 energy + AG force parity and wall timing.
 * Source: hen/pbs/out/parity_nacl_18/ladder_wigner_fix/ (jobs 8728551, 8728585).
 */

type LadderRow = {
  W: number;
  tiles: number;
  backend: string;
  job: string;
  E: number;
  Epa: number;
  Fmax: number;
  Frms: number;
  agFdOk: boolean;
  maxAgFd: number;
  meanAgFd: number;
  dE_meV_atom: number | null;
  maxdF: number | null;
  rmsdF: number | null;
  cos: number | null;
  elapsed_s: number;
};

type Spot = {
  W: number;
  atom: number;
  comp: string;
  AG: number;
  FD: number;
  abs: number;
};

const ROWS: LadderRow[] = [
  {
    W: 1,
    tiles: 1,
    backend: "none (single-tile)",
    job: "8728551",
    E: -157578.5311152171,
    Epa: -3.377454799280202,
    Fmax: 0.7191007066769073,
    Frms: 0.13551183168267367,
    agFdOk: true,
    maxAgFd: 2.204111805403919e-6,
    meanAgFd: 1.1547820026803274e-6,
    dE_meV_atom: null,
    maxdF: null,
    rmsdF: null,
    cos: null,
    elapsed_s: 876.623880221001,
  },
  {
    W: 2,
    tiles: 2,
    backend: "xccl",
    job: "8728585",
    E: -157578.53111521716,
    Epa: -3.377454799280203,
    Fmax: 0.7191007066769073,
    Frms: 0.1355118316826737,
    agFdOk: true,
    maxAgFd: 1.1853419064900006e-6,
    meanAgFd: 5.14212842744271e-7,
    dE_meV_atom: -1.2475921835019592e-12,
    maxdF: 1.0547118733938987e-15,
    rmsdF: 1.8756068404630046e-16,
    cos: 1.0,
    elapsed_s: 363.29490853700554,
  },
  {
    W: 4,
    tiles: 4,
    backend: "xccl",
    job: "8728585",
    E: -157578.53111521725,
    Epa: -3.3774547992802053,
    Fmax: 0.7191007066769075,
    Frms: 0.1355118316826737,
    agFdOk: true,
    maxAgFd: 5.608879208585105e-7,
    meanAgFd: 2.847455368441523e-7,
    dE_meV_atom: -3.1189804587548977e-12,
    maxdF: 9.71445146547012e-16,
    rmsdF: 1.953680488948065e-16,
    cos: 0.9999999999999998,
    elapsed_s: 140.79972898797132,
  },
  {
    W: 6,
    tiles: 6,
    backend: "xccl",
    job: "8728585",
    E: -157578.53111521725,
    Epa: -3.3774547992802053,
    Fmax: 0.7191007066769072,
    Frms: 0.1355118316826737,
    agFdOk: true,
    maxAgFd: 6.864857607091768e-7,
    meanAgFd: 2.813442964641477e-7,
    dE_meV_atom: -3.1189804587548977e-12,
    maxdF: 1.0547118733938987e-15,
    rmsdF: 1.967899623181044e-16,
    cos: 1.0,
    elapsed_s: 102.81694451300427,
  },
  {
    W: 12,
    tiles: 12,
    backend: "xccl",
    job: "8728585",
    E: -157578.5311152171,
    Epa: -3.377454799280202,
    Fmax: 0.7191007066769071,
    Frms: 0.13551183168267367,
    agFdOk: true,
    maxAgFd: 3.669428735308955e-7,
    meanAgFd: 1.778737018841138e-7,
    dE_meV_atom: 0.0,
    maxdF: 1.0998146837692957e-15,
    rmsdF: 2.0770887789434963e-16,
    cos: 1.0,
    elapsed_s: 71.84824613598175,
  },
];

const SPOTS: Spot[] = [
  { W: 1, atom: 0, comp: "x", AG: -0.09562206883260027, FD: -0.09562427294440567, abs: 2.204111805403919e-6 },
  { W: 1, atom: 0, comp: "y", AG: 0.1049932775601474, FD: 0.10499352356418967, abs: 2.4600404227581585e-7 },
  { W: 1, atom: 0, comp: "z", AG: -0.05189418936303954, FD: -0.051894166972488165, abs: 2.2390551378259627e-8 },
  { W: 1, atom: 23328, comp: "x", AG: -0.10060518849670534, FD: -0.10060422937385738, abs: 9.591228479627345e-7 },
  { W: 1, atom: 23328, comp: "y", AG: -0.06511226152255328, FD: -0.06511443643830717, abs: 2.1749157538830666e-6 },
  { W: 1, atom: 23328, comp: "z", AG: -0.022382454655011146, FD: -0.022384338080883026, abs: 1.8834258718805619e-6 },
  { W: 1, atom: 46655, comp: "x", AG: 0.04548111674265848, FD: 0.04548172000795603, abs: 6.032652975496156e-7 },
  { W: 1, atom: 46655, comp: "y", AG: 0.04433161253287485, FD: 0.044330517994239926, abs: 1.0945386349214825e-6 },
  { W: 1, atom: 46655, comp: "z", AG: 0.17022136053212641, FD: 0.17022015526890755, abs: 1.2052632188674917e-6 },
  { W: 2, atom: 0, comp: "x", AG: -0.09562206883260033, FD: -0.0956222356762737, abs: 1.6684367337704842e-7 },
  { W: 2, atom: 0, comp: "y", AG: 0.10499327756014717, FD: 0.10499250493012369, abs: 7.726300234878192e-7 },
  { W: 2, atom: 0, comp: "z", AG: -0.05189418936303925, FD: -0.05189504008740187, abs: 8.50724362615185e-7 },
  { W: 2, atom: 23328, comp: "x", AG: -0.10060518849670516, FD: -0.1006049569696188, abs: 2.3152708636398067e-7 },
  { W: 2, atom: 23328, comp: "y", AG: -0.06511226152255334, FD: -0.06511283572763205, abs: 5.742050787072017e-7 },
  { W: 2, atom: 23328, comp: "z", AG: -0.02238245465501132, FD: -0.022382300812751055, abs: 1.5384226026426973e-7 },
  { W: 2, atom: 46655, comp: "x", AG: 0.04548111674265867, FD: 0.04548230208456516, abs: 1.1853419064900006e-6 },
  { W: 2, atom: 46655, comp: "y", AG: 0.044331612532874536, FD: 0.04433197318576276, abs: 3.606528882274529e-7 },
  { W: 2, atom: 46655, comp: "z", AG: 0.17022136053212641, FD: 0.17022102838382125, abs: 3.3214830516548055e-7 },
  { W: 4, atom: 0, comp: "x", AG: -0.09562206883260017, FD: -0.09562194463796914, abs: 1.2419463103763295e-7 },
  { W: 4, atom: 0, comp: "y", AG: 0.10499327756014724, FD: 0.10499352356418967, abs: 2.460040424284715e-7 },
  { W: 4, atom: 0, comp: "z", AG: -0.051894189363039446, FD: -0.05189402145333588, abs: 1.6790970356478363e-7 },
  { W: 4, atom: 23328, comp: "x", AG: -0.1006051884967051, FD: -0.10060553904622793, abs: 3.5054952282620455e-7 },
  { W: 4, atom: 23328, comp: "y", AG: -0.06511226152255352, FD: -0.06511254468932748, abs: 2.8316677395945344e-7 },
  { W: 4, atom: 23328, comp: "z", AG: -0.022382454655011406, FD: -0.022382737370207906, abs: 2.8271519649999965e-7 },
  { W: 4, atom: 46655, comp: "x", AG: 0.04548111674265862, FD: 0.04548055585473776, abs: 5.608879208585105e-7 },
  { W: 4, atom: 46655, comp: "y", AG: 0.044331612532875, FD: 0.04433182766661048, abs: 2.1513373547887849e-7 },
  { W: 4, atom: 46655, comp: "z", AG: 0.1702213605321262, FD: 0.17022102838382125, abs: 3.3214830494343595e-7 },
  { W: 6, atom: 0, comp: "x", AG: -0.09562206883260031, FD: -0.09562209015712142, abs: 2.1324521107257688e-8 },
  { W: 6, atom: 0, comp: "y", AG: 0.10499327756014734, FD: 0.10499308700673282, abs: 1.9055341451967855e-7 },
  { W: 6, atom: 0, comp: "z", AG: -0.05189418936303958, FD: -0.05189358489587903, abs: 6.044671605476282e-7 },
  { W: 6, atom: 23328, comp: "x", AG: -0.10060518849670526, FD: -0.10060539352707565, abs: 2.0503037038988037e-7 },
  { W: 6, atom: 23328, comp: "y", AG: -0.06511226152255324, FD: -0.06511210813187063, abs: 1.5339068261399635e-7 },
  { W: 6, atom: 23328, comp: "z", AG: -0.02238245465501129, FD: -0.022382737370207906, abs: 2.827151966144914e-7 },
  { W: 6, atom: 46655, comp: "x", AG: 0.04548111674265862, FD: 0.04548142896965146, abs: 3.1222699284350064e-7 },
  { W: 6, atom: 46655, comp: "y", AG: 0.044331612532874744, FD: 0.04433153662830591, abs: 7.590456883171948e-8 },
  { W: 6, atom: 46655, comp: "z", AG: 0.17022136053212653, FD: 0.17022204701788723, abs: 6.864857607091768e-7 },
  { W: 12, atom: 0, comp: "x", AG: -0.09562206883260028, FD: -0.0956222356762737, abs: 1.6684367341868178e-7 },
  { W: 12, atom: 0, comp: "y", AG: 0.10499327756014717, FD: 0.10499308700673282, abs: 1.905534143531451e-7 },
  { W: 12, atom: 0, comp: "z", AG: -0.0518941893630395, FD: -0.05189402145333588, abs: 1.6790970362029478e-7 },
  { W: 12, atom: 23328, comp: "x", AG: -0.10060518849670518, FD: -0.10060524800792336, abs: 5.9511218189478576e-8 },
  { W: 12, atom: 23328, comp: "y", AG: -0.06511226152255357, FD: -0.0651123991701752, abs: 1.3764762162027377e-7 },
  { W: 12, atom: 23328, comp: "z", AG: -0.022382454655011486, FD: -0.022382737370207906, abs: 2.827151964202024e-7 },
  { W: 12, atom: 46655, comp: "x", AG: 0.045481116742658534, FD: 0.04548099241219461, abs: 1.243304639242382e-7 },
  { W: 12, atom: 46655, comp: "y", AG: 0.044331612532874876, FD: 0.044331245590001345, abs: 3.669428735308955e-7 },
  { W: 12, atom: 46655, comp: "z", AG: 0.17022136053212622, FD: 0.1702214649412781, abs: 1.0440915187981403e-7 },
];

const WIDTHS = [1, 2, 4, 6, 12] as const;
const E0 = ROWS[0].elapsed_s;

function sci(x: number | null | undefined, dig = 3): string {
  if (x === null || x === undefined) return "—";
  if (x === 0) return "0";
  return x.toExponential(dig);
}

function fmtE(x: number): string {
  return x.toFixed(8);
}

export default function Nacl18W112Analysis() {
  const [wSel, setWSel] = useCanvasState<number>("spotW", 1);

  const parityRows = ROWS.map((r) => [
    String(r.W),
    r.backend,
    fmtE(r.E),
    r.Epa.toFixed(8),
    r.Fmax.toFixed(6),
    r.agFdOk ? "PASS" : "FAIL",
    sci(r.maxAgFd),
    sci(r.dE_meV_atom),
    sci(r.maxdF),
    r.cos === null ? "—" : r.cos.toFixed(10),
  ]);

  const timingRows = ROWS.map((r) => {
    const speedup = E0 / r.elapsed_s;
    return [
      String(r.W),
      r.elapsed_s.toFixed(1),
      speedup.toFixed(2) + "×",
      ((100 * speedup) / r.W).toFixed(0) + "%",
      r.job,
    ];
  });

  const elapsedChart = ROWS.map((r) => ({
    label: `W=${r.W}`,
    value: Number(r.elapsed_s.toFixed(1)),
  }));

  const speedupChart = ROWS.map((r) => ({
    label: `W=${r.W}`,
    value: Number((E0 / r.elapsed_s).toFixed(2)),
  }));

  const spotRows = SPOTS.filter((s) => s.W === wSel).map((s) => [
    String(s.atom),
    s.comp,
    s.AG.toFixed(10),
    s.FD.toFixed(10),
    sci(s.abs, 4),
  ]);

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>NaCl 18³ — W=1…12 energy, AG forces, timing</H1>
        <Text tone="secondary" size="small">
          Rocksalt 18×18×18 · 46 656 atoms · a=5.64 Å · rattle=0.05 · seed=0 ·
          FP64 UMA · Wigner-prep chunk on. Source:
          pbs/out/parity_nacl_18/ladder_wigner_fix/ · W=1 job 8728551 · W=2…12
          job 8728585 (1-node xccl).
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="5/5" label="AG≡FD PASS" tone="success" />
        <Stat value="5/5" label="E+F vs W=1 PASS" tone="success" />
        <Stat value="12.2×" label="elapsed W1→W12" />
        <Stat value="≤2.2e−6" label="max |AG−FD| (eV/Å)" tone="success" />
      </Grid>

      <Callout tone="warning" title="Timing definition">
        Ladder elapsed_s = calc/Ray setup + E₀ + full AG forces + 9 spot FD
        energies. It is not warm ef_mean and does not split energy-only vs
        AG-only. Superlinear “efficiency” vs W is expected because FD energy
        evals also scale with W.
      </Callout>

      <Card>
        <CardHeader trailing={<Pill active size="sm">parity</Pill>}>
          Energy and AG force parity (vs W=1)
        </CardHeader>
        <CardBody>
          <Table
            framed
            striped
            stickyHeader
            headers={[
              "W",
              "backend",
              "E (eV)",
              "E/atom",
              "Fmax AG",
              "AG≡FD",
              "max|AG−FD|",
              "ΔE meV/atom",
              "max|ΔF| vs W1",
              "cos F vs W1",
            ]}
            columnAlign={[
              "right",
              "left",
              "right",
              "right",
              "right",
              "left",
              "right",
              "right",
              "right",
              "right",
            ]}
            rowTone={["success", "success", "success", "success", "success"]}
            rows={parityRows}
          />
          <Text tone="tertiary" size="small">
            Bars: |ΔE|/N ≪ 1e−6 meV/atom; max|ΔF| ~1e−15 eV/Å (FP noise);
            cos(F,F_W1)=1. AG≡FD tol 1e−5 eV/Å on atoms 0, 23328, 46655 × xyz.
          </Text>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill active size="sm">wall</Pill>}>
          Wall timing (setup + E + AG + 9×FD)
        </CardHeader>
        <CardBody>
          <Table
            framed
            striped
            headers={["W", "elapsed_s", "vs W=1", "elapsed/W %", "job"]}
            columnAlign={["right", "right", "right", "right", "left"]}
            rows={timingRows}
          />
          <Text tone="tertiary" size="small">
            elapsed/W % = 100 × (elapsed_W1 / elapsed_W) / W — not inference
            parallel efficiency.
          </Text>
        </CardBody>
      </Card>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>elapsed_s by W</CardHeader>
          <CardBody>
            <BarChart
              categories={elapsedChart.map((d) => d.label)}
              series={[{ name: "elapsed_s", data: elapsedChart.map((d) => d.value) }]}
              height={220}
              valueSuffix=" s"
              showValues
            />
            <Text tone="tertiary" size="small">
              Y: wall seconds (setup+E+AG+FD) · Source: ladder_wigner_fix summary.json
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Wall speedup vs W=1</CardHeader>
          <CardBody>
            <BarChart
              categories={speedupChart.map((d) => d.label)}
              series={[
                { name: "elapsed_W1 / elapsed_W", data: speedupChart.map((d) => d.value) },
              ]}
              height={220}
              valueSuffix="×"
              showValues
            />
            <Text tone="tertiary" size="small">
              Y: speedup vs W=1 · Ideal linear = W; W=12 ≈ 12.2× on this fat timer
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="info" title="No split E / AG timers on this ladder">
        The probe does not record separate energy-only or AG-force wall times.
        Ancillary (different jobs / not parity-linked): W=1 single E+F ≈ 19.5 s
        (sweep_nacl_w1_memory/n18); W=12 warm ef_mean ≈ 2.47 s
        (w12_n_memory_sweep_16_40/n18, Phase1 on — energy differs by ~0.017
        meV/atom from this ladder). Do not mix those with the parity table.
      </Callout>

      <Divider />

      <Stack gap={10}>
        <H2>Spot AG vs FD (eV/Å)</H2>
        <Text tone="secondary" size="small">
          Central difference eps=1e−4 on three atoms × xyz.
        </Text>
        <Row gap={8} wrap>
          {WIDTHS.map((w) => (
            <span key={w}>
              <Pill active={wSel === w} onClick={() => setWSel(w)}>
                {`W=${w}`}
              </Pill>
            </span>
          ))}
        </Row>
        <Card>
          <CardHeader>
            <H3>{`W=${wSel}`}</H3>
          </CardHeader>
          <CardBody>
            <Table
              framed
              striped
              headers={["atom", "comp", "AG", "FD", "|AG−FD|"]}
              columnAlign={["right", "left", "right", "right", "right"]}
              rows={spotRows}
            />
          </CardBody>
        </Card>
      </Stack>
    </Stack>
  );
}
