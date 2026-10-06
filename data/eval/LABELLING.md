# Labelling guidelines (frozen before labelling)

These rules were written and frozen (hash in `LABELLING.freeze.json`) **before** any row of
`credit_to_label.csv` or `multilingual_to_label.csv` was read for labelling. They are not
changed afterwards. All examples below are invented for this guide; none come from either set.

**Who labels and how.** The labels in `*_labelled.csv` are **AI-annotated** (the assistant
that built this project), not human-labelled:
- **Blind.** The labeller sees only the row id and the original headline. Model outputs, machine
  translations, linked entities, weak-label rules, keyword lists and the selection bucket are
  not in view.
- **Original language.** Each headline is read in its original language.
- **Two passes, three steps.** Two independent passes run in different shuffled orders, each
  label with a one-line rationale. Cohen's kappa is computed between the passes. Disagreements
  are settled in a third (adjudication) pass and flagged `adjudicated = yes`.
- **Spot check.** A human checks 20 random rows per set (`spot_check.csv`).

The headline is judged **as written**. Don't guess beyond the text and don't use what happened
later.

## Event label (one per headline)

Pick the label for the **main thing that happened or is reported**. If two apply, use the
precedence rule at the end.

### Credit Event
A borrower's (company, bank, sovereign, sub-sovereign) ability or willingness to pay its debt
**worsens or fails**, or a formal credit/insolvency process starts. This includes:
- a default, a missed payment or coupon, or a distressed exchange;
- bankruptcy, insolvency, administration, receivership or liquidation filings (NCLT/IBC
  admission in India);
- a debt restructuring forced by distress;
- a **rating downgrade**, a negative outlook, or a negative watch;
- a covenant breach or waiver request;
- a bank run or a bailout of an insolvent lender.

Positive examples:
1. "Retailer X files for Chapter 11 protection after talks with lenders fail"
2. "Moody's cuts Country Y to Caa1, citing rising default risk"
3. "NCLT admits insolvency plea against Z Infra over unpaid dues"

Negative examples:
1. "Fitch affirms Bank A at 'A-'; outlook stable" → Other/Non-event (no deterioration)
2. "S&P upgrades Country B to BBB on fiscal gains" → Macroeconomic (improvement, not a credit event)
3. "Credit card spending rose 6% in March" → Macroeconomic (the word "credit" is not a credit event)

Edge cases:
- **Affirmations, upgrades, and stable or positive outlooks** are never Credit Event. Rating
  news with no deterioration is Macroeconomic for a sovereign and Other/Non-event for a company.
- **Tax credits, carbon credits, credit cards, and credit or loan growth** are not credit events.
  Label the underlying topic: policy or tax is Macroeconomic or Regulatory/Legal; carbon is
  ESG/Climate.
- **Household or personal debt** (a farmer's loans, a celebrity's debts) is not a Credit Event
  unless an institution or sovereign is in default. Label it Other/Non-event, or
  Macroeconomic if it is aggregate.
- **Talk of risk is not a credit event.** "Fears of default", "could default" or "warns of
  downgrade" counts only if a rating agency acts (negative outlook or watch counts) or the
  borrower itself says it cannot pay.
- **A completed, agreed refinancing** without distress is Other/Non-event. A distressed exchange
  or a restructuring forced by inability to pay is a Credit Event.
- **Fraud or money-laundering charges against a lender** are Regulatory/Legal, unless the
  headline says the lender failed or defaulted.

### Regulatory/Legal
Actions by regulators, courts, prosecutors or legislators aimed at a firm or sector:
- fines, penalties, licences, probes, lawsuits, rulings and settlements;
- AML or money-laundering findings;
- new rules for an industry.

Positive examples:
1. "Regulator fines Bank C $40 m for anti-money-laundering failures"
2. "Court blocks Company D's patent claim"
3. "SEC opens probe into accounting at E Corp"

Negative examples:
1. "Central bank raises policy rate by 25 bp" → Macroeconomic
2. "Government imposes sanctions on Country F" → Geopolitical
3. "Company G appoints new general counsel" → Management Change

Edge cases:
- **Country-level standing.** An AML grey-list or FATF status change for a country is
  Regulatory/Legal.
- **Antitrust.** Merger reviews and antitrust rulings on a deal are Regulatory/Legal when the
  ruling is the news.
- **Tax policy.** General tax-policy changes are Macroeconomic. A tax dispute with a named firm
  is Regulatory/Legal.

### Macroeconomic
Economy-wide or market-wide conditions:
- rates, inflation, GDP, jobs, trade data, budgets and public debt levels;
- currencies, commodity prices and broad market moves;
- sovereign rating **upgrades** or affirmations;
- aggregate credit or loan growth.

Positive examples:
1. "Inflation slows to 3.1% in August"
2. "Rupee hits record low against dollar"
3. "Bank lending growth accelerates to 15%"

Negative examples:
1. "Country H defaults on Eurobond coupon" → Credit Event
2. "Oil major I reports record profit" → Earnings/Guidance
3. "Troops mass on border" → Geopolitical

Edge cases:
- **Index moves.** "Stocks fall", "Sensex rises" and similar index moves are Macroeconomic.
- **Single-stock moves.** A single stock moving without a stated cause is Other/Non-event.

### Geopolitical
War, military action, terrorism, sanctions, tariffs and trade wars, diplomacy, elections,
coups, and political crises.

Positive examples:
1. "Missiles strike port city overnight"
2. "EU adopts new sanctions package against Country J"
3. "Prime minister resigns after losing confidence vote"

Negative examples:
1. "Government cuts fuel tax" → Macroeconomic
2. "Court rules against ministry in land case" → Regulatory/Legal
3. "Election-year budget raises spending" → Macroeconomic

Edge cases:
- **Tariffs** are Geopolitical when they are part of a trade conflict.
- **Domestic crime and policing** with no political dimension is Other/Non-event.

### Merger/Acquisition
Deals to buy, sell, merge or take stakes in companies or assets, including bids, takeovers,
divestments, IPOs and buy-outs.

Positive examples: "K agrees to buy L for $2 bn"; "M sells its cement unit"; "N to list
on stock exchange in IPO".

Negative examples:
1. "O signs supply contract with P" → Operational/Supply Chain or Other
2. "Regulator blocks Q's merger" → Regulatory/Legal
3. "R raises $50 m in bond sale" → Other/Non-event

### Earnings/Guidance
Company results, profit, revenue, guidance and forecasts, dividends, and analyst estimate
changes about results.

Positive examples: "S posts 20% rise in quarterly profit"; "T cuts full-year outlook";
"U declares interim dividend".

Negative examples:
1. "V's shares jump 5%" (no cause) → Other/Non-event
2. "W misses bond payment" → Credit Event
3. "GDP growth beats forecast" → Macroeconomic

### Management Change
Appointment, resignation or removal of executives or board members.

Positive examples: "X names new CEO"; "Y's CFO resigns"; "Board ousts Z's chairman".

Negative examples:
1. "Minister replaced in cabinet reshuffle" → Geopolitical
2. "Company hires 500 engineers" → Other/Non-event
3. "Central bank governor speaks on inflation" → Macroeconomic

### Product Launch
A new product, service or model is launched or approved for sale, including drug approvals.

Positive examples: "AA unveils new smartphone"; "BB gets approval for diabetes drug";
"CC launches app in India".

Negative examples:
1. "DD recalls 1 m cars" → Operational/Supply Chain
2. "EE reports strong iPhone sales" → Earnings/Guidance
3. "Trade fair opens in city" → Other/Non-event

### Operational/Supply Chain
Disruptions or changes to physical operations:
- strikes, recalls, plant shutdowns, accidents and outages;
- shortages, logistics problems, capacity expansions, and big contracts or orders.

Positive examples: "Workers strike at FF's plant"; "GG recalls 200,000 vehicles"; "Port
congestion delays shipments".

Negative examples:
1. "HH hacked, customer data stolen" → Cyber/Technology Risk
2. "Floods kill 30" → ESG/Climate
3. "II buys rival" → Merger/Acquisition

### ESG/Climate
Environment, climate, natural disasters, emissions, carbon markets and carbon credits,
sustainability, and governance or social controversies framed as ESG.

Positive examples: "Heatwave cuts power output"; "JJ sets 2040 net-zero goal"; "Carbon
credit prices hit record".

Negative examples:
1. "Solar tax credit extended by Congress" → Macroeconomic
2. "Oil price rises" → Macroeconomic
3. "KK fined for pollution" → Regulatory/Legal

### Cyber/Technology Risk
Cyberattacks, data breaches, ransomware, major IT outages, and AI or tech failures that harm
firms or users.

Positive examples: "Ransomware attack halts LL's operations"; "Data of 10 m MM customers
leaked"; "Global IT outage grounds flights".

Negative examples:
1. "NN launches AI chip" → Product Launch
2. "Tech stocks rally" → Macroeconomic
3. "Regulator fines OO over data rules" → Regulatory/Legal

### Other/Non-event
Anything that fits none of the above:
- sport, culture, lifestyle and local crime;
- general features, opinion pieces and listicles;
- single-stock moves without a cause;
- personal finance and consumer advice (credit card tips);
- ratings affirmations of companies.

Positive examples: "Ten ways to boost your credit score"; "Local team wins cup final";
"PP's shares gain 2%".

Negative examples: any headline that fits a class above.

### Precedence when two classes apply
1. Credit Event
2. Geopolitical
3. Regulatory/Legal
4. Merger/Acquisition
5. Earnings/Guidance
6. Management Change
7. Cyber/Technology Risk
8. Operational/Supply Chain
9. Product Launch
10. ESG/Climate
11. Macroeconomic
12. Other/Non-event

In words: a credit deterioration outranks the context it happens in, and a specific
corporate event outranks a general macro backdrop. For example, "Sanctions push Bank QQ into
default" is a Credit Event.

## Sentiment label (multilingual set only)

Sentiment is judged from the headline text alone, for the **main financial subject** of the
headline: the company, country or market it is about. Ask what a financial reader holding
that subject would feel.

**positive**: good news for the subject: growth, profit, an upgrade, a win, easing risk.
Examples:
1. "Exports rise 12%"
2. "RR wins $1 bn order"
3. "Inflation eases"

**negative**: bad news for the subject: losses, cuts, defaults, conflict, disasters, fines,
falling prices for the subject's assets.
Examples:
1. "SS shares plunge on fraud probe"
2. "Floods destroy crops"
3. "Currency slides to record low"

**neutral**: factual, mixed or non-financial content with no clear good or bad direction.
Examples:
1. "Board meeting scheduled for Friday"
2. "Minister visits trade fair"
3. "Football club signs striker"

Edge cases:
- **Mixed headlines** with no dominant direction are neutral.
- **War, terrorism and disasters** are negative for the affected country or market.
- **Policy rate moves** are neutral unless the headline frames them as good or bad.
- **Non-financial human-interest stories** are neutral, even when sad (a local accident with
  no economic angle).
