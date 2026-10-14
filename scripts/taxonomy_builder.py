import json

TERMS_BY_CATEGORY = {
    "property_type": [
        "single family residence", "single family home", "townhouse", "townhome",
        "condominium", "condo", "duplex", "triplex", "fourplex",
        "manufactured home", "mobile home", "single story", "single level",
        "two story", "2 story", "three story", "split level", "detached home",
        "attached home", "end unit", "new construction", "custom home",
        "guest house", "casita", "accessory dwelling unit", "adu", "detached adu",
        "multi-generational", "cabin", "penthouse", "ranch style", "mediterranean",
        "spanish style", "craftsman", "contemporary", "mid-century", "farmhouse",
        "cottage", "bungalow", "country home", "tudor",
    ],
    "interior_features": [
        "stainless steel appliances", "granite countertops", "quartz countertops",
        "marble countertops", "granite", "quartz", "kitchen island", "center island",
        "oversized island", "breakfast bar", "gourmet kitchen", "chef's kitchen",
        "walk-in pantry", "butler's pantry", "custom cabinetry", "soft-close",
        "tile backsplash", "backsplash", "farmhouse sink", "gas range",
        "double oven", "range hood", "washer and dryer", "hardwood flooring",
        "hardwood floors", "laminate flooring", "luxury vinyl plank",
        "vinyl plank flooring", "tile flooring", "wall-to-wall carpet",
        "crown molding", "recessed lighting", "vaulted ceilings", "high ceilings",
        "soaring ceilings", "beamed ceiling", "floor-to-ceiling", "light-filled",
        "natural light", "fireplace", "gas fireplace", "built-in cabinetry",
        "built-ins", "soaking tub", "walk-in shower", "spa-inspired", "spa-like",
        "dual vanity", "dual vanities", "skylight", "skylights",
        "plantation shutters", "wet bar", "wine cellar", "french doors",
        "sliding glass doors", "ceiling fans",
    ],
    "layout_rooms": [
        "open floor plan", "open concept", "split floor plan", "great room",
        "family room", "living room", "formal living room", "formal dining",
        "formal dining room", "dining room", "dining area", "breakfast nook",
        "primary suite", "primary bedroom", "primary bathroom", "master suite",
        "master bedroom", "en-suite", "ensuite", "jack and jill bathroom",
        "walk-in closet", "walk-in closets", "home office", "den", "bonus room",
        "loft", "game room", "media room", "home theater", "playroom", "home gym",
        "in-law suite", "guest suite", "main level", "main level bedroom",
        "downstairs bedroom", "laundry room", "mudroom", "pantry", "basement",
        "finished basement", "attic", "sunroom", "flex space", "powder room",
        "half bath", "balcony",
    ],
    "outdoor_lot": [
        "pool", "swimming pool", "private pool", "above ground pool",
        "infinity pool", "pool and spa", "spa", "hot tub", "jacuzzi", "backyard",
        "backyard oasis", "front yard", "patio", "covered patio", "deck",
        "rooftop deck", "fire pit", "bbq", "built-in bbq", "outdoor kitchen",
        "outdoor living", "indoor-outdoor living", "outdoor dining",
        "outdoor entertaining", "outdoor space", "gazebo", "pergola", "fenced",
        "fenced yard", "large lot", "corner lot", "cul-de-sac", "flat lot",
        "hillside lot", "greenbelt", "mountain views", "ocean views", "city views",
        "city lights", "sunset views", "valley views", "golf course views",
        "canyon views", "lake views", "panoramic views", "fruit trees",
        "mature trees", "landscaping", "drought-tolerant landscaping",
        "artificial turf", "turf", "courtyard", "garden", "horse property",
        "acreage", "curb appeal",
    ],
    "systems_parking_energy": [
        "attached garage", "detached garage", "two-car garage", "2 car garage",
        "three-car garage", "3 car garage", "oversized garage", "tandem garage",
        "carport", "rv parking", "ev charger", "ev charging",
        "electric vehicle charging", "driveway", "off-street parking",
        "parking space", "guest parking", "garage storage", "solar",
        "solar panels", "solar system", "owned solar", "paid off solar",
        "leased solar", "battery backup", "central air", "air conditioning",
        "hvac", "central heating", "new hvac", "tankless water heater",
        "new water heater", "new furnace", "dual pane", "dual pane windows",
        "energy efficient", "smart home", "smart thermostat", "whole house fan",
        "copper plumbing", "upgraded electrical panel", "water softener",
        "security system", "elevator", "septic",
    ],
    "condition_updates": [
        "turnkey", "move-in ready", "move-in condition", "meticulously maintained",
        "well maintained", "beautifully maintained", "beautifully updated",
        "beautifully remodeled", "immaculate", "immaculate condition", "pristine",
        "recently renovated", "fully renovated", "renovated", "remodeled",
        "fully remodeled", "completely remodeled", "newly remodeled",
        "remodeled kitchen", "updated kitchen", "upgraded kitchen",
        "updated bathrooms", "updated throughout", "modern updates", "new roof",
        "new flooring", "new carpet", "new paint", "new interior paint",
        "fresh paint", "fresh interior paint", "freshly painted", "exterior paint",
        "new appliances", "new windows", "new plumbing", "new fixtures",
        "brand new", "newly built", "model home", "like new", "upgraded",
        "needs tlc", "fixer-upper", "investor special", "as-is", "needs work",
        "recent upgrades", "low-maintenance",
    ],
    "location_community": [
        "gated", "gated community", "guard-gated", "gated entry",
        "golf course community", "golf course", "clubhouse", "resort-style",
        "resort-style amenities", "community pool", "community amenities",
        "fitness center", "tennis courts", "pickleball courts", "playground",
        "walking trails", "hiking trails", "trails", "dog park", "open space",
        "near shopping", "close to shopping", "shopping and dining",
        "near restaurants", "near schools", "close to schools",
        "top-rated schools", "award-winning schools", "school district",
        "walking distance", "minutes from", "freeway access", "major freeways",
        "commuter routes", "public transportation", "walkable", "beach access",
        "quiet street", "quiet neighborhood", "family-friendly neighborhood",
        "master-planned community", "55+ community", "senior community",
        "lake community", "wine country", "waterfront", "24-hour security",
        "near university", "near parks",
    ],
    "financing_ownership": [
        "hoa", "hoa fees", "hoa dues", "low hoa", "no hoa", "mello-roos",
        "special assessment", "property taxes", "low property taxes",
        "seller financing", "owner carry", "assumable loan", "fha", "va loan",
        "cash only", "1031 exchange", "investor", "investment property",
        "rental income", "income property", "income-producing", "rental potential",
        "tenant occupied", "vacant", "vacation rental", "vacation rental license",
        "short-term rental", "airbnb", "short sale", "reo", "probate",
        "motivated seller", "price improvement", "price reduced", "lender",
    ],
}

terms = []
counter = 1
for category, term_list in TERMS_BY_CATEGORY.items():
    prefix = category[:4]
    for term in term_list:
        terms.append({
            "id": f"{prefix}_{counter:03d}",
            "term": term,
            "category": category,
        })
        counter += 1

with open('data/processed/taxonomy.json', 'w') as f:
    json.dump({"terms": terms}, f, indent=2)

print(f"Wrote {len(terms)} terms across {len(TERMS_BY_CATEGORY)} categories")
for category, term_list in TERMS_BY_CATEGORY.items():
    print(f"  {category}: {len(term_list)}")
