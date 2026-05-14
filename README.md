# NHL Trade Evaluator

A data-driven trade analysis tool that quantifies what each team gains and loses in any proposed NHL trade.

Built with real hockey analytics methodology — not vibes.

---

## What it does

Given a proposed trade (players + draft picks per side), the evaluator answers:

- **Who wins the trade and by how much?**
- **Which team gets the win-now edge?**
- **Which team gets the better future value?**

Every player gets a composite trade value score built from three components:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| WAR estimate | 40% | Current season impact above replacement |
| Contract efficiency | 25% | WAR produced per $1M cap hit |
| Peak WAR projection | 25% | Age-curve adjusted future value |
| Control years | 10% | Years of team control remaining |

Draft picks are valued using a curve fitted to 20 years of historical NHL draft outcomes — not a borrowed football chart.

---

## Methodology

### Player valuation

We train a Ridge regression model predicting WAR from even-strength NST metrics:

- `xGF/60` — offensive contribution independent of linemates
- `xGA/60` — defensive contribution
- `CF% Rel` — relative possession impact
- `iXG` — individual shot quality
- `OZS%` — deployment context
- `TOI` — proxy for coach trust / role

Training targets are Evolving Hockey WAR estimates. Using Ridge over OLS because xGF and CF% are correlated; Ridge handles this without overfitting.

### Pick value model

Exponential decay curve fitted to career WAR outcomes for every NHL draft pick 2000–2020. Only uses mature draft classes (5+ years of NHL data available). Outputs expected WAR and P(NHL impact) for any pick 1–224.

Why not the Jimmy Johnson chart? NFL draft value charts don't translate to hockey. NHL draft variance is significantly higher, ELC contracts make early picks disproportionately valuable, and trades involve players + retained salary — not just picks.

### Age curve

Comparables-based projection: classify player as ascending / peak / declining relative to known position-specific peak ages (F: 25.5, D: 26.5), then apply a modest linear adjustment. Conservative by design — we don't want the model making wild 10-year projections.

---

## Data sources

| Source | Data | Update cadence |
|--------|------|---------------|
| [Natural Stat Trick](https://www.naturalstattrick.com) | Skater stats (xGF, CF%, TOI, etc.) | Daily during season |
| [Evolving Hockey](https://www.evolving-hockey.com) | WAR / GSVA estimates | Daily during season |
| [Hockey Reference](https://www.hockey-reference.com) | Historical draft data | Weekly |
| [PuckPedia](https://puckpedia.com) | Contract data | Weekly |

**Note:** This project scrapes publicly available data respectfully — rate limited to ≤10 requests/minute with aggressive caching. Please respect these sites; they are small operations that provide enormous value to the hockey community.

---

## Project structure

```
nhl_trade_evaluator/
├── data/
│   ├── database.py          # SQLAlchemy schema
│   ├── scheduler.py         # Automated update scheduler
│   └── manual_imports/      # Fallback CSVs if scrapers blocked
├── scrapers/
│   ├── base.py              # Rate limiting + caching base class
│   ├── nst_scraper.py       # Natural Stat Trick
│   ├── eh_scraper.py        # Evolving Hockey
│   └── hockeyref_scraper.py # Hockey Reference draft data
├── models/
│   ├── pick_value.py        # Draft pick valuation model
│   ├── player_valuation.py  # Player WAR + trade value model
│   └── trade_evaluator.py   # Trade verdict engine
├── api/
│   └── main.py              # FastAPI application
└── frontend/
    └── index.html           # Trade evaluator UI
```

---

## Setup

```bash
git clone https://github.com/yourusername/nhl-trade-evaluator
cd nhl-trade-evaluator
pip install -r requirements.txt

# Initialize database
python -m nhl_trade_evaluator.data.database

# Scrape historical draft data (run once)
python -c "
from nhl_trade_evaluator.scrapers.hockeyref_scraper import HockeyRefDraftScraper
scraper = HockeyRefDraftScraper()
picks = scraper.scrape_range(2000, 2020)
print(f'Scraped {len(picks)} draft picks')
"

# Scrape current season stats
python -m nhl_trade_evaluator.data.scheduler --once

# Train models
python -c "
from nhl_trade_evaluator.data.database import get_session
from nhl_trade_evaluator.models.pick_value import build_pick_value_model
session = get_session()
model = build_pick_value_model(session)
print('Pick value model trained')
"

# Start API
uvicorn nhl_trade_evaluator.api.main:app --reload --port 8000

# Open frontend
open frontend/index.html
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model readiness status |
| GET | `/players/search?q=` | Player autocomplete |
| GET | `/players/{name}/value` | Single player valuation |
| GET | `/picks/value?overall=` | Pick value by position |
| GET | `/picks/table` | Full pick value curve |
| POST | `/trades/evaluate` | Full trade evaluation |

---

## Limitations and known issues

- Age is not yet stored in the contract table — age curve projections use position defaults until this is added
- Evolving Hockey scraper may require manual CSV fallback if their site blocks automated requests
- Model trained on EV stats only — special teams deployment is a known gap
- Goalies not yet supported

---

## Author

Built as part of a hockey analytics portfolio while working toward a career in NHL hockey operations.

Feedback from anyone in the hockey analytics community is genuinely welcome.
