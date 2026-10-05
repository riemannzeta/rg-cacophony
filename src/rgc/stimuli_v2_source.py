"""PREREG-2 part B stimuli: two cue slots before the target, graded and conflicting fields.

Frames 0-7 are train, 8-15 test. Cue phrases per (word, sense): indices 0-3 are used only in
train frames, 4-7 only in test frames (disjoint, as in v1). Every content word of a cue phrase is a
cue word. No content word may appear in that word's v1 cue phrases (checked by the builder).
Neutral fillers are shared across words and must carry no sense information.
"""

FRAMES = [
    # train (0-7)
    "Between {c1} and {c2}, my cousin could only think about the",
    "What with {c1} and {c2}, nobody stopped asking about the",
    "After {c1}, and later {c2}, the whole family wondered about the",
    "Both {c1} and {c2} came up again when we talked about the",
    "First {c1}, then {c2}; in the end it all came back to the",
    "With {c1} on Monday and {c2} on Tuesday, my neighbour kept mentioning the",
    "Despite {c1} and {c2}, the old woman would only talk about the",
    "Somewhere between {c1} and {c2}, my father started worrying about the",
    # test (8-15)
    "Alongside {c1} and {c2}, the speaker brought up the",
    "Once {c1} and {c2} had both been mentioned, someone asked about the",
    "Amid {c1} and {c2}, my roommate kept bringing up the",
    "Thinking of {c1} and {c2}, the boy asked his mother about the",
    "Right after {c1} and just before {c2}, they started discussing the",
    "Mixed in with {c1} and {c2}, there was a question about the",
    "Following {c1} and {c2}, the committee spent an hour on the",
    "In a story about {c1} and {c2}, the narrator described the",
]

FILLERS = [
    "the long weekend", "the bus ride", "the phone call", "the rainy afternoon",
    "the slow morning", "the short chat", "the quiet evening", "the family reunion",
]

# word -> {sense: [8 cue phrases]}; sense order matches v1 (A, B).
CUES = {
    "bank": {
        "river": ["the beaver dam", "the cattails", "the bridge pilings", "the angler",
                  "the marsh", "the tadpoles", "the ferry crossing", "the mossy boulders"],
        "money": ["the tax refund", "the overdue invoice", "the paycheck", "the currency exchange",
                  "the stock portfolio", "the inheritance", "the lottery winnings", "the bitcoin purchase"],
    },
    "bat": {
        "animal": ["the stalactites", "the spelunking club", "the abandoned tunnel", "the Halloween decorations",
                   "the bug zapper", "the moonlit sky", "the church steeple", "the cobwebs"],
        "sport": ["the curveball", "the line drive", "the bleachers", "the pinch runner",
                  "the grand slam", "the infield", "the bunt", "the outfield"],
    },
    "pitcher": {
        "baseball": ["the splitter", "the earned run average", "the Cy Young award", "the spitball",
                     "the relief appearance", "the pickoff move", "the radar gun", "the rosin bag"],
        "jug": ["the margaritas", "the ice cubes", "the stoneware", "the gravy boat",
                "the spout", "the mojitos", "the dishwasher", "the kombucha"],
    },
    "seal": {
        "animal": ["the sardines", "the tidal pools", "the trawler", "the herring",
                   "the narwhals", "the frozen fjord", "the mackerel", "the puffins"],
        "closure": ["the rubber washer", "the shrink wrap", "the childproof cap", "the pressure cooker",
                    "the monogrammed ring", "the wedding invitations", "the thermos", "the pharaoh's tomb"],
    },
    "spring": {
        "season": ["the vernal pools", "the hay fever", "the pollen count", "the baby chicks",
                   "the daylight saving", "the hyacinths", "the magnolias", "the ladybugs"],
        "coil": ["the shock absorbers", "the diving board", "the mousetrap", "the jack-in-the-box",
                 "the garage door", "the pinball plunger", "the door hinge", "the trigger assembly"],
    },
    "match": {
        "contest": ["the badminton doubles", "the squash court", "the fencing duel", "the hockey rink",
                    "the cup qualifier", "the tiebreak", "the hat trick", "the yellow card"],
        "fire": ["the oil lamp", "the gunpowder", "the incense", "the pipe tobacco",
                 "the kerosene", "the flare", "the ashtray", "the propane tank"],
    },
    "mole": {
        "animal": ["the grubs", "the golf course greens", "the velvety fur", "the spade-like claws",
                   "the vole", "the gopher", "the rodent repellent", "the backyard trenches"],
        "skin": ["the tanning bed", "the pathology lab", "the cosmetic surgeon", "the UV exposure",
                 "the beauty mark", "the cryotherapy", "the stitches", "the laser removal"],
    },
    "pen": {
        "writing": ["the crossword puzzle", "the thank-you cards", "the felt-tip markers", "the poetry contest",
                    "the sketchbook", "the exam answers", "the cursive practice", "the doodles"],
        "enclosure": ["the petting zoo", "the turkeys", "the alpacas", "the manure",
                      "the water trough", "the county fair", "the goslings", "the straw bedding"],
    },
    "bark": {
        "tree": ["the cinnamon", "the termites", "the aspen", "the chestnut",
                 "the eucalyptus", "the arborist", "the cork", "the willow"],
        "dog": ["the German shepherd", "the dachshund", "the squirrels", "the intruder",
                "the muzzle", "the rottweiler", "the pug", "the chew toy"],
    },
    "palm": {
        "hand": ["the high five", "the paper cut", "the tarot cards", "the pinky",
                 "the carpal tunnel", "the fist", "the wedding ring", "the applause"],
        "tree": ["the Los Angeles skyline", "the Sahara", "the pineapple", "the camels",
                 "the Polynesian luau", "the Mediterranean villa", "the tiki torches", "the bananas"],
    },
}

# Field conditions: (slot1, slot2) in {"A", "B", "N"}; h = #A - #B.
CONDITIONS = ["AA", "BB", "AN", "NA", "BN", "NB", "AB", "BA", "NN"]
