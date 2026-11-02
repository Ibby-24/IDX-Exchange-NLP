# Week 6 signal labeling guidelines

These rules were written **before** any listing was labeled and before `signal_extractor.py` existed. A signal is labeled **present** when the remark itself says it, about *this* property or its immediate setting. Absence of a mention is "not present" (the remarks say nothing about most things, so recall is measured against what the text states, not against the truth about the house).

General rules
1. Read the raw remark. Label what a careful reader would conclude from the words, not what the house probably has.
2. Negated, hypothetical or "could be" statements do not count ("no HOA" is not a financing term; "could be converted to an ADU" is not a feature; "ideal for a home office" is not an office).
3. Community-level facts count only for the signals defined as community signals (`gated_community`, `golf_course`, `senior_55plus`, `parks_open_space`).
4. One mention is enough. A signal is a yes/no per listing, not a count.
5. Do not label from the listing's price, city or any column. Text only.

## Condition (6)

| Signal | Label it when the remark says | Not this |
|---|---|---|
| `renovated_updated` | the home was remodeled, renovated, updated or upgraded (kitchen, baths, "throughout", "fully upgraded") | a single new item (that is `new_components`); "ready for your updates" |
| `new_construction` | newly built, new construction, brand new home, never lived in, builder/tract model of a new community | "brand new kitchen" in an old house |
| `new_components` | specific new or recently replaced parts: new roof, flooring, carpet, paint, "freshly painted", appliances, windows, HVAC, water heater, plumbing, electrical, fence; "newer roof" | general "updated" |
| `move_in_ready` | turnkey, move-in ready / condition, ready to move in | "ready for new owner" alone |
| `well_maintained` | well / meticulously / beautifully maintained, pristine, immaculate, pride of ownership, lovingly cared for | "clean" alone |
| `needs_work` | fixer, needs TLC / repairs / updating / rehab, as-is, handyman or investor or contractor special, sweat equity, original condition (not updated), "bring your vision" for a house in need of work, blank canvas to remodel | "make it your own" on an updated home |

## Financing and ownership (13 signals + 2 fields)

| Signal | Label it when the remark says |
|---|---|
| `seller_financing` | seller financing / carry, owner financing / carry, "seller will carry" |
| `assumable_loan` | assumable loan or loan assumption (including "assumable 2.5%") |
| `fha` | FHA offered, welcome, considered, approved, or the complex is FHA approved |
| `va` | VA loan / VA approved / VA financing |
| `cash_only` | cash only, cash buyers only, cash offers |
| `exchange_1031` | 1031 exchange |
| `short_sale` | short sale |
| `reo` | REO, bank owned, foreclosure |
| `probate_estate` | probate, estate sale, trust sale, conservatorship |
| `tenant_occupied` | tenant occupied, currently leased / rented, tenants in place (vacant is not labeled) |
| `investment_income` | the remark pitches it as an investment or income source: investor, investment property, rental income, income-producing, cap rate, "live in one rent the other" |
| `seller_credit` | seller credit or concession, closing-cost credit, rate buy-down, "credit toward closing" |
| `motivated_seller` | motivated seller, priced to sell, price reduced / improvement / lowered, bring all offers |

Fields (one value per listing; blank when not mentioned)
- `hoa`: `none` ("no HOA", "no association", "not in an HOA"), `low` ("low HOA", "very low HOA", a clearly small stated amount called low), `present` (an HOA, association dues or fees are mentioned as applying to this property and are not called low or none).
- `mello_roos`: `none` ("no Mello-Roos"), `present` (Mello-Roos tax mentioned without "no").

## Location (14)

| Signal | Label it when the remark says |
|---|---|
| `gated_community` | gated, guard-gated, gated entry or community |
| `golf_course` | on, backing to or near a golf course, golf community, golf membership |
| `near_schools` | schools are near or walkable, or school quality is a selling point ("top-rated schools", a named school with "close to"). A bare district name does not count |
| `near_shopping_dining` | near, close to or walkable to shopping, dining, restaurants, groceries, entertainment |
| `freeway_commute` | freeway or highway access, commuter routes, easy commute |
| `public_transit` | BART, Metro, Metrolink, train, bus, transit |
| `walkable` | "walkable", "walking distance", "walk to" any amenity, "short walk" |
| `near_beach_water` | beach, ocean, lake, bay, marina nearby or accessible (minutes to the beach, steps from the sand, lake access) |
| `waterfront` | the property itself is on water: waterfront, oceanfront, lakefront, canal-front, own boat dock |
| `views` | a named view: mountain, ocean, city, city lights, valley, canyon, lake, hills, golf course |
| `quiet_street` | quiet / peaceful / tranquil street or neighborhood, cul-de-sac, secluded, private road |
| `parks_open_space` | parks, trails, greenbelt, open space nearby or backing to the home |
| `senior_55plus` | 55+, active adult, senior or retirement community |
| `near_downtown` | close to or in the heart of a downtown or city center, or minutes from a named downtown |
