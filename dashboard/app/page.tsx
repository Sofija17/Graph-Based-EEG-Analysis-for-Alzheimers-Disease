'use client';

import {
  Activity,
  BrainCircuit,
  FlaskConical,
  Network,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  CircleHelp,
  Lightbulb,
  Target,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, XAxis, YAxis } from 'recharts';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';

const methodData = [
  {
    method: 'Pearson',
    accuracy: 75.4,
    sensitivity: 77.8,
    specificity: 72.4,
    f1: 77.8,
    auc: 74.9,
  },
  {
    method: 'Spearman',
    accuracy: 76.9,
    sensitivity: 77.8,
    specificity: 75.9,
    f1: 78.9,
    auc: 74.7,
  },
  {
    method: 'Coherence',
    accuracy: 78.5,
    sensitivity: 69.4,
    specificity: 89.7,
    f1: 78.1,
    auc: 74.8,
  },
];
const folds = [
  {
    fold: 1,
    epoch: 14,
    threshold: 0.582,
    accuracy: 0.615,
    f1: 0.667,
    auc: 0.667,
  },
  {
    fold: 2,
    epoch: 1,
    threshold: 0.534,
    accuracy: 0.923,
    f1: 0.933,
    auc: 0.975,
  },
  {
    fold: 3,
    epoch: 4,
    threshold: 0.658,
    accuracy: 0.769,
    f1: 0.769,
    auc: 0.929,
  },
  {
    fold: 4,
    epoch: 15,
    threshold: 0.564,
    accuracy: 0.769,
    f1: 0.727,
    auc: 0.714,
  },
  {
    fold: 5,
    epoch: 50,
    threshold: 0.503,
    accuracy: 0.692,
    f1: 0.778,
    auc: 0.762,
  },
];
const spectral = [
  ['O2', 'Однос тета / алфа', '3.178', '1.447', '+0.718', '6.29×10⁻⁵'],
  ['T5', 'Однос тета / алфа', '3.376', '1.607', '+0.713', '6.29×10⁻⁵'],
  ['O2', 'Релативна алфа-моќност', '0.032', '0.080', '−0.680', '1.28×10⁻⁴'],
  ['T5', 'Релативна алфа-моќност', '0.029', '0.070', '−0.661', '1.55×10⁻⁴'],
  ['T6', 'Однос тета / алфа', '3.308', '1.676', '+0.657', '1.55×10⁻⁴'],
  ['O2', 'Однос бавни / брзи', '51.959', '26.878', '+0.653', '1.55×10⁻⁴'],
];
const connectivity = [
  ['O2', 'Тежинско кластерирање', '0.640', '0.411', '+0.697', '4.80×10⁻⁵'],
  ['O2', 'Јачина на јазол', '4.738', '2.680', '+0.688', '4.80×10⁻⁵'],
  ['T6', 'Јачина на јазол', '4.692', '3.184', '+0.594', '5.63×10⁻⁴'],
  ['F7', 'Тежинско кластерирање', '0.450', '0.633', '−0.582', '5.63×10⁻⁴'],
  ['F7', 'Јачина на јазол', '3.202', '5.132', '−0.546', '1.24×10⁻³'],
  ['F4', 'Јачина на јазол', '4.927', '6.483', '−0.533', '1.54×10⁻³'],
];
const chartConfig = {
  accuracy: { label: 'Точност', color: '#16a085' },
  sensitivity: { label: 'Сензитивност', color: '#e09f3e' },
  specificity: { label: 'Специфичност', color: '#355070' },
  auc: { label: 'ROC–AUC', color: '#9b5de5' },
};
const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

function EvidenceTable({ rows }: { rows: string[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200">
      <table className="w-full min-w-[680px] text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {[
              'Канал',
              'Биомаркер',
              'AD просек',
              'CN просек',
              'Ефект',
              'FDR q',
            ].map((x) => (
              <th key={x} className="px-4 py-3 font-semibold">
                {x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row) => (
            <tr key={`${row[0]}-${row[1]}`} className="hover:bg-teal-50/40">
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={`px-4 py-3 ${j === 0 ? 'font-semibold text-slate-900' : ''} ${j === 4 ? (cell.startsWith('+') ? 'text-teal-700' : 'text-blue-700') : ''}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-white/8 bg-[#071a24]/95 text-white backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3 lg:px-8">
          <div className="flex items-center gap-3">
            <span className="grid size-9 place-items-center rounded-xl bg-teal-400/15 text-teal-300 ring-1 ring-teal-300/25">
              <BrainCircuit className="size-5" />
            </span>
            <div>
              <p className="text-sm font-semibold">EEG–AD истражување</p>
              <p className="text-[11px] text-slate-400">
                Контролна табла на ниво на испитаник
              </p>
            </div>
          </div>
          <nav className="hidden gap-5 text-xs text-slate-300 md:flex">
            <a href="#overview">За проектот</a>
            <a href="#performance">Перформанси</a>
            <a href="#biomarkers">Биомаркери</a>
            <a href="#validation">Валидација</a>
            <a href="#methodology">Методологија</a>
          </nav>
          <span className="rounded-full bg-emerald-400/10 px-3 py-1.5 text-xs font-medium text-emerald-300 ring-1 ring-emerald-300/20">
            Анализата е завршена
          </span>
        </div>
      </header>

      <section className="border-b bg-[#0b2531] text-white">
        <div className="mx-auto max-w-7xl px-5 py-12 lg:px-8">
          <div className="max-w-3xl">
            <p className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-teal-300">
              <FlaskConical className="size-4" />
              Граф-базирана EEG анализа
            </p>
            <h1 className="text-3xl font-semibold leading-tight tracking-tight sm:text-5xl">
              Граф-базирана анализа и класификација на Alzheimer’s disease од
              resting-state EEG функционална конективност
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
              Проектот проверува дали EEG сигналите и мозочните
              врски се разликуваат кај здрави лица и лица со Алцхајмерова болест,
              и дали компјутерски модел може да ги разликува двете групи.
            </p>
          </div>
          <div className="mt-9 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ['Испитаници', '65', '36 AD · 29 CN', BrainCircuit],
              ['EEG графови', '12.826', '19 канали · 4 опсези', Network],
              ['Најдобра точност', '78,5%', 'GCN со кохерентност', Activity],
              [
                'Pearson OOF AUC',
                '0.749',
                '65 независни предвидувања',
                ShieldCheck,
              ],
            ].map(([label, value, detail, Icon]) => (
              <div
                key={String(label)}
                className="rounded-2xl border border-white/10 bg-white/[.055] p-5"
              >
                <div className="flex items-center justify-between text-slate-400">
                  <span className="text-xs font-medium uppercase tracking-wider">
                    {String(label)}
                  </span>
                  {typeof Icon !== 'string' && (
                    <Icon className="size-4 text-teal-300" />
                  )}
                </div>
                <div className="mt-3 text-3xl font-semibold tabular-nums">
                  {String(value)}
                </div>
                <p className="mt-1 text-sm text-slate-400">{String(detail)}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-7xl space-y-10 px-5 py-10 lg:px-8">
        <section id="overview" className="scroll-mt-24 space-y-6">
          <div>
            <p className="eyebrow">За проектот</p>
            <h2>Што истражуваме и зошто?</h2>
            <p className="section-copy max-w-4xl">
              EEG ја мери електричната активност на мозокот преку електроди
              поставени на главата. Во проектот, секој EEG канал го претставуваме
              како точка, односно јазол во граф. Врската меѓу два канала покажува
              колку слично работат тие две мозочни области. Така EEG снимката се
              претвора во мозочна мрежа што може да ја анализира GCN модел.
            </p>
          </div>

          <div className="grid gap-5 lg:grid-cols-3">
            <Card className="border-teal-200 bg-teal-50/60">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CircleHelp className="size-5 text-teal-700" />
                  Истражувачко прашање
                </CardTitle>
              </CardHeader>
              <CardContent className="leading-7 text-slate-700">
                Дали qEEG карактеристиките и функционалната поврзаност се
                различни кај здравите испитаници и испитаниците со Алцхајмерова
                болест? Дали GNN модел може да ги разликува двете групи?
              </CardContent>
            </Card>
            <Card className="border-blue-200 bg-blue-50/60">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="size-5 text-blue-700" />
                  Цел на проектот
                </CardTitle>
              </CardHeader>
              <CardContent className="leading-7 text-slate-700">
                Да се откријат можни EEG биомаркери за Алцхајмерова болест преку
                анализа на мозочните ритми и врските меѓу EEG каналите, а потоа
                тие информации да се употребат за автоматска класификација.
              </CardContent>
            </Card>
            <Card className="border-amber-200 bg-amber-50/60">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lightbulb className="size-5 text-amber-700" />
                  Краток одговор
                </CardTitle>
              </CardHeader>
              <CardContent className="leading-7 text-slate-700">
                <strong>Да.</strong> Најдени се разлики во мозочните ритми и во
                одредени регионални врски. GCN моделот ги разликува групите со
                најдобра точност од <strong>78,5%</strong>, но резултатот сè уште
                не е доволен за клиничка дијагноза.
              </CardContent>
            </Card>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Најважни поими, едноставно објаснети</h3>
            <div className="mt-4 grid gap-4 text-sm leading-6 text-slate-700 md:grid-cols-2 lg:grid-cols-4">
              <p><strong>qEEG карактеристика:</strong> број што кажува колку е застапен одреден мозочен ритам, како алфа или тета.</p>
              <p><strong>Функционална поврзаност:</strong> мера за тоа колку синхронизирано се менуваат сигналите од два EEG канала.</p>
              <p><strong>Граф:</strong> мрежа во која каналите се јазли, а нивната поврзаност е претставена со ребра.</p>
              <p><strong>Биомаркер:</strong> мерлива особина што може да укажува на разлика поврзана со болеста.</p>
            </div>
            <div className="mt-5 border-t border-slate-200 pt-5">
              <h3 className="text-base font-semibold text-slate-900">Како да ги разберете мерките за успешност?</h3>
              <div className="mt-3 grid gap-4 text-sm leading-6 text-slate-700 md:grid-cols-2 lg:grid-cols-4">
                <p><strong>Точност:</strong> колкав процент од сите испитаници е правилно класифициран.</p>
                <p><strong>Сензитивност:</strong> колкав процент од лицата со AD моделот правилно ги препознал.</p>
                <p><strong>Специфичност:</strong> колкав процент од здравите лица моделот правилно ги препознал.</p>
                <p><strong>ROC–AUC:</strong> општа способност за раздвојување на групите; 0,5 е случајно, а 1,0 е совршено.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="performance" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Перформанси на моделот</p>
            <h2>Споредба на методите за поврзаност</h2>
            <p className="section-copy">
              Секој метод користи исти поделби на испитаниците, исти почетни
              вредности за случајните операции, исти карактеристики на јазлите
              и исти GCN хиперпараметри.
            </p>
          </div>
          <div className="grid gap-6 lg:grid-cols-[1.7fr_1fr]">
            <Card className="border-0 shadow-sm ring-slate-200">
              <CardHeader>
                <CardTitle>Резултати од вкрстената валидација</CardTitle>
                <CardDescription>Резултатите се прикажани во проценти</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={chartConfig}
                  className="h-[340px] w-full aspect-auto"
                >
                  <BarChart data={methodData}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="method" tickLine={false} axisLine={false} />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tickLine={false}
                      axisLine={false}
                    />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Legend />
                    <Bar
                      dataKey="accuracy"
                      fill="var(--color-accuracy)"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="sensitivity"
                      fill="var(--color-sensitivity)"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="specificity"
                      fill="var(--color-specificity)"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="auc"
                      fill="var(--color-auc)"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ChartContainer>
                <div className="mt-5 rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-700">
                  <p><strong>Како се чита графикот:</strong> долу се трите начини
                  на создавање врски меѓу EEG каналите. Висината на столбовите го
                  покажува процентот на успешност; повисок столб значи подобар резултат.</p>
                  <p className="mt-2"><strong>Што покажува:</strong> кохерентноста
                  има највисока точност (78,5%) и најдобро ги препознава здравите
                  лица (89,7%), но пропушта повеќе лица со AD. Pearson и Spearman
                  имаат порамномерна сензитивност и специфичност.</p>
                </div>
              </CardContent>
            </Card>
            <Card className="border-0 bg-teal-950 text-white shadow-sm ring-teal-900">
              <CardHeader>
                <CardTitle className="text-xl">Толкување</CardTitle>
                <CardDescription className="text-teal-100/65">
                  Што потврдува споредбата
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5 leading-6 text-teal-50/85">
                <p>
                  Способноста за разликување на AD од CN е речиси иста: ROC–AUC,
                  пресметан врз сите тест-предвидувања, е{' '}
                  <strong>0,747–0,749</strong> кај сите методи.
                </p>
                <p>
                  Кохерентноста ја зголемува специфичноста на <strong>89,7%</strong>,
                  но ја намалува AD сензитивноста на <strong>69,4%</strong>.
                </p>
                <div className="rounded-xl bg-white/8 p-4 text-sm ring-1 ring-white/10">
                  <strong className="text-teal-200">Заклучок:</strong> повисоката
                  точност кај кохерентноста не докажува општо подобар класификатор.
                  Спектралните карактеристики на јазлите носат постабилен сигнал.
                </div>
              </CardContent>
            </Card>
          </div>
          <div className="mt-6 grid gap-5 md:grid-cols-3">
            <figure className="figure-card">
              <img
                src="/results/oof_roc_curve.png"
                alt="Pearson ROC крива од вкрстената валидација"
              />
              <figcaption>
                ROC крива на ниво на испитаник · референтен Pearson модел
              </figcaption>
              <div className="px-4 pb-4 text-sm leading-6 text-slate-600">
                <strong>Што гледаме?</strong> Сината линија ја покажува способноста
                на моделот да ги разликува AD и CN при различни прагови. Сивата
                дијагонала е случајно погодување. Сината линија е над неа, а
                ROC–AUC од 0,749 значи умерено добра, но не совршена поделба.
              </div>
            </figure>
            <figure className="figure-card">
              <img
                src="/results/oof_confusion_matrix.png"
                alt="Pearson матрица на конфузија"
              />
              <figcaption>21 точни CN, 28 точни AD, вкупно 16 грешки</figcaption>
              <div className="px-4 pb-4 text-sm leading-6 text-slate-600">
                <strong>Како се чита?</strong> Редовите се вистинската група, а
                колоните се предвидувањето на моделот. Точно се препознаени 21
                здрав и 28 AD испитаници. Погрешно се класифицирани 8 здрави како
                AD и 8 AD како здрави — вкупно 49 од 65 се точни.
              </div>
            </figure>
            <figure className="figure-card">
              <img
                src="/results/oof_probability_distribution.png"
                alt="Предвидени AD веројатности по група"
              />
              <figcaption>
                Распределба на AD веројатностите од тест-предвидувањата
              </figcaption>
              <div className="px-4 pb-4 text-sm leading-6 text-slate-600">
                <strong>Како се чита?</strong> Секоја точка е еден испитаник.
                Повисока точка значи дека моделот дал поголема веројатност за AD.
                Портокаловите AD точки најчесто се повисоко од сините CN точки,
                но има преклопување; токму таму настануваат грешките.
              </div>
            </figure>
          </div>
        </section>

        <section id="biomarkers" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Биомаркерски докази</p>
            <h2>Групни разлики на ниво на испитаник</h2>
            <p className="section-copy">
              Секој пациент придонесува со една агрегирана вредност. Тестовите
              користат двостран Mann–Whitney U, ранг-бисериски големини на ефект
              и Benjamini–Hochberg FDR корекција.
            </p>
          </div>
          <div className="space-y-8">
            <div>
              <h3 className="mb-4 text-lg font-semibold text-slate-900">Спектрална qEEG анализа</h3>
              <div className="grid gap-5 lg:grid-cols-[1.45fr_1fr]">
                <Card>
                  <CardHeader>
                    <CardTitle>Најсилни спектрални наоди</CardTitle>
                    <CardDescription>
                      78 од 133 тестови канал–карактеристика се значајни по FDR
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <EvidenceTable rows={spectral} />
                    <div className="mt-4 space-y-2 text-sm leading-6 text-slate-600">
                      <p><strong>Што значат колоните?</strong> „AD просек“ и „CN
                      просек“ се просечните вредности во двете групи. Позитивен
                      ефект значи повисока вредност кај AD, а негативен ефект —
                      пониска вредност кај AD.</p>
                      <p><strong>Главен резултат:</strong> лицата со AD имаат
                      повисок однос тета/алфа и пониска релативна алфа-моќност,
                      особено во задниот дел на главата кај O2 и T5.</p>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Просторна шема</CardTitle>
                    <CardDescription>
                      Алфа-моќноста е пониска кај AD, најизразено постериорно
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      className="mx-auto max-h-[430px] object-contain"
                      src="/results/topomap_alpha.png"
                      alt="Мапа на скалпот со ефектите на релативната алфа-моќност"
                    />
                    <div className="mt-4 text-sm leading-6 text-slate-600">
                      <strong>Како се чита мапата?</strong> Кругот е поглед на
                      главата одозгора: носот е горе, а задниот дел е долу. Сината
                      боја значи дека алфа-моќноста е пониска кај AD. Потемно сино
                      значи поголема разлика. Заокружените електроди имаат
                      статистички потврдена разлика по FDR корекцијата.
                    </div>
                  </CardContent>
                </Card>
              </div>
              <div className="mt-5 grid gap-5 lg:grid-cols-2">
                <figure className="figure-card">
                  <img
                    src="/results/topomap_theta_alpha.png"
                    alt="Мапа на скалпот за односот тета–алфа"
                  />
                  <figcaption>
                    Односот тета/алфа е доследно повисок кај AD
                  </figcaption>
                  <div className="px-4 pb-4 text-sm leading-6 text-slate-600">
                    <strong>Значење:</strong> црвените области покажуваат дека кај
                    AD има релативно повеќе бавна тета активност во однос на алфа
                    активноста. Ова е најизразено во задните EEG канали.
                  </div>
                </figure>
                <figure className="figure-card">
                  <img
                    src="/results/heatmap_theta_alpha.png"
                    alt="Топлинска мапа по канали за односот тета–алфа"
                  />
                  <figcaption>
                    Позитивни ефекти низ сите 19 канали; најсилни кај O2 и T5
                  </figcaption>
                  <div className="px-4 pb-4 text-sm leading-6 text-slate-600">
                    <strong>Како се чита?</strong> Секоја колона е EEG канал.
                    Потопла боја и поголем позитивен број значат повисок однос
                    тета/алфа кај AD. Ѕвездичка означува дека разликата останува
                    значајна и по корекција за повеќекратни тестирања.
                  </div>
                </figure>
              </div>
            </div>
            <div>
              <h3 className="mb-4 text-lg font-semibold text-slate-900">Биомаркери на поврзаност</h3>
              <div className="grid gap-5 lg:grid-cols-[1.45fr_1fr]">
                <Card>
                  <CardHeader>
                    <CardTitle>Најсилни наоди за поврзаноста</CardTitle>
                    <CardDescription>
                      23 од 43 тестови се значајни по FDR
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <EvidenceTable rows={connectivity} />
                    <div className="mt-4 space-y-2 text-sm leading-6 text-slate-600">
                      <p><strong>Јачина на јазол</strong> кажува колку силно е еден
                      EEG канал поврзан со останатите. <strong>Тежинско
                      кластерирање</strong> кажува дали соседните канали формираат
                      тесно поврзана локална група.</p>
                      <p><strong>Главен резултат:</strong> најсилните разлики се
                      регионални — особено кај O2, T6, F7 и F4 — а не еднакво
                      распоредени низ целиот мозок.</p>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Регионално, не глобално</CardTitle>
                    <CardDescription>
                      Ефект на јачината на јазолот по канал
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src="/results/connectivity_node_strength.png"
                      alt="Ефекти на јачината на јазолот по EEG канал"
                    />
                    <div className="mt-4 text-sm leading-6 text-slate-600">
                      <strong>Како се чита графикот?</strong> Секоја колона е еден
                      EEG канал. Црвено и позитивен број значат поголема јачина на
                      врските кај AD; сино и негативен број значат поголема јачина
                      кај CN. Ѕвездичката означува статистички значајна разлика.
                      Најсилен позитивен ефект има O2 (+0,69), а најсилен негативен
                      ефект F7 (−0,55).
                    </div>
                  </CardContent>
                </Card>
              </div>
              <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-950">
                <div className="flex gap-3">
                  <AlertTriangle className="mt-0.5 size-5 shrink-0 text-amber-600" />
                  <p>
                    <strong>Толкувајте внимателно.</strong> Pearson анализата на
                    биомаркерите за поврзаност користи апсолутни тежини на ребрата
                    со праг на највисоките 30%. Таа открива регионални разлики во
                    јачината и топологијата на врските, но не може да разликува
                    позитивни од негативни корелации во оригиналниот референтен
                    збир на податоци.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="validation" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Детали за вкрстената валидација</p>
            <h2>Резултати на Pearson моделот по поделба</h2>
            <p className="section-copy">
              За секоја поделба, прагот е избран само според валидациските
              податоци со индексот Youden J, а потоа е применет врз испитаниците
              издвоени за тестирање.
            </p>
          </div>
          <Card>
            <CardContent className="pt-1">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs uppercase tracking-wide text-slate-500">
                      {[
                        'Поделба',
                        'Најдобра епоха',
                        'Праг',
                        'Точност',
                        'F1',
                        'ROC–AUC',
                        'Тест-испитаници',
                      ].map((x) => (
                        <th key={x} className="px-4 py-3">
                          {x}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {folds.map((x) => (
                      <tr
                        key={x.fold}
                        className="border-b border-slate-100 last:border-0"
                      >
                        <td className="px-4 py-3 font-semibold">{x.fold}</td>
                        <td className="px-4 py-3">{x.epoch}</td>
                        <td className="px-4 py-3 font-mono">
                          {x.threshold.toFixed(3)}
                        </td>
                        <td className="px-4 py-3">{pct(x.accuracy)}</td>
                        <td className="px-4 py-3">{x.f1.toFixed(3)}</td>
                        <td className="px-4 py-3">{x.auc.toFixed(3)}</td>
                        <td className="px-4 py-3">13</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
          <div className="mt-4 rounded-xl border border-blue-200 bg-blue-50 p-5 text-sm leading-6 text-blue-950">
            <strong>Зошто има пет поделби?</strong> Испитаниците се делат на пет
            делови. Моделот петпати се тренира одново: секојпат учи од четири
            делови, а се тестира на преостанатиот дел. Така секој испитаник е
            тестиран еднаш со модел што претходно не ги видел неговите податоци.
            Разликите меѓу редовите покажуваат дека резултатот зависи и од тоа кои
            лица се нашле во конкретната тест-група.
          </div>
          <div className="mt-5 grid gap-4 sm:grid-cols-3">
            <div className="mini-callout">
              <span>Просечна точност</span>
              <strong>0.754 ± 0.102</strong>
            </div>
            <div className="mini-callout">
              <span>Просечен F1 резултат</span>
              <strong>0.775 ± 0.088</strong>
            </div>
            <div className="mini-callout">
              <span>Просечен ROC–AUC</span>
              <strong>0.809 ± 0.121</strong>
            </div>
          </div>
        </section>

        <section id="methodology" className="scroll-mt-24">
          <div className="mb-5">
            <p className="eyebrow">Методологија</p>
            <h2>Од EEG снимка до научен доказ</h2>
          </div>
          <div className="grid gap-4 md:grid-cols-5">
            {[
              [
                '01',
                'Претпроцесирање',
                '0,5–45 Hz, просечна референца, епохи од 4 секунди',
              ],
              [
                '02',
                'Карактеристики на јазол',
                'Релативна делта, тета, алфа и бета моќност',
              ],
              [
                '03',
                'Ребра на графот',
                'Најсилни 30% Pearson, Spearman или кохерентни врски',
              ],
              [
                '04',
                'GCN по испитаник',
                'Просечна AD веројатност низ сите епохи на пациентот',
              ],
              [
                '05',
                'Евалуација',
                'Петкратна валидација по испитаник, без мешање податоци меѓу поделбите',
              ],
            ].map(([n, t, d]) => (
              <div className="step-card" key={n}>
                <span>{n}</span>
                <h3>{t}</h3>
                <p>{d}</p>
              </div>
            ))}
          </div>
          <div className="mt-6 grid gap-5 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="size-5 text-teal-600" />
                  Применети заштитни мерки
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="clean-list">
                  <li>Сите епохи од еден пациент остануваат во иста поделба</li>
                  <li>При тренирањето, секој испитаник има еднаква тежина</li>
                  <li>Рано запирање според валидациската загуба</li>
                  <li>Прагот за одлука е избран без тест-податоци</li>
                  <li>Секој испитаник добива точно едно тест-предвидување од модел што не бил трениран со неговите податоци</li>
                  <li>FDR корекција за повеќекратни биомаркерски тестови</li>
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="size-5 text-amber-600" />
                  Ограничувања
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="clean-list">
                  <li>Мал примерок од еден збир на податоци: 65 испитаници</li>
                  <li>Нема валидација на надворешна кохорта</li>
                  <li>Односите може да се нестабилни при многу мала алфа-моќност</li>
                  <li>Прагот за поврзаност е фиксиран на најсилните 30%</li>
                  <li>
                    Тековните резултати се истражувачки, а не клинички
                    дијагностички систем
                  </li>
                  <li>
                    Основниот модел сè уште треба да се оцени со истата
                    петкратна вкрстена валидација
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="rounded-3xl bg-[#0b2531] p-7 text-white sm:p-10">
          <p className="eyebrow text-teal-300">Општ заклучок</p>
          <h2 className="max-w-3xl text-white">
            Репродуцибилен сигнал, со јасни ограничувања
          </h2>
          <p className="mt-4 max-w-4xl text-base leading-7 text-slate-300">
            GCN на ниво на испитаник ги раздвојува AD и CN подобро од случајност,
            додека постериорното намалување на алфа-моќноста и зголемениот однос
            тета/алфа ги даваат најјасните qEEG докази. Разликите во поврзаноста
            се регионално информативни, особено околу O2, T5/T6 и F7, но промената
            на конструкцијата на ребрата не го подобрува значително ROC–AUC.
            Резултатите оправдуваат понатамошна валидација, не клиничка примена.
          </p>
          <div className="mt-7 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl bg-white/[.07] p-5 ring-1 ring-white/10">
              <h3 className="font-semibold text-teal-200">1. Има ли qEEG разлики?</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Да. Кај AD се забележува помала алфа активност и поголем однос
                тета/алфа, најмногу во задните канали O1, O2, T5 и T6.
              </p>
            </div>
            <div className="rounded-2xl bg-white/[.07] p-5 ring-1 ring-white/10">
              <h3 className="font-semibold text-teal-200">2. Има ли разлики во поврзаноста?</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Да, но тие се повеќе регионални отколку глобални. Најизразени се
                околу O2, T5/T6, F7 и F4.
              </p>
            </div>
            <div className="rounded-2xl bg-white/[.07] p-5 ring-1 ring-white/10">
              <h3 className="font-semibold text-teal-200">3. Може ли GCN да ги разликува групите?</h3>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Да, со умерена успешност. Најдобрата точност е 78,5%, што е
                подобро од случајно погодување, но не е доволно за клиничка употреба.
              </p>
            </div>
          </div>
        </section>
      </div>
      <footer className="border-t bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-5 py-6 text-xs text-slate-500 sm:flex-row sm:justify-between lg:px-8">
          <span>Граф-базирана EEG анализа за Алцхајмерова болест</span>
          <span>
            Создадено од проверени проектни резултати · 65 набљудувања на ниво
            на испитаник
          </span>
        </div>
      </footer>
    </main>
  );
}
