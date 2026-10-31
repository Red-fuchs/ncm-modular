NEWS = [
 "The Scottish government has announced new funding to help rural communities cope with rising energy costs this winter.",
 "A man has been arrested after a car crashed into a shop in the centre of Cardiff on Tuesday evening.",
 "Manchester United came from behind to beat Everton 3-1 at Old Trafford, with Marcus Bellamy scoring twice.",
 "The Bank of England has held interest rates at 5.25% for the fifth time in a row, with inflation still above the 2% target.",
 "Researchers at the University of Oxford say a new blood test could detect several types of cancer years before symptoms appear.",
 "Police in Belfast are appealing for witnesses after a woman in her 60s was injured in a hit-and-run on the Lisburn Road.",
 "A Welsh farmer has been fined 12,000 pounds after cattle escaped onto the A470 near Brecon last summer.",
 "Celtic have signed striker Kenji Furuta on a four-year deal from Vissel Kobe for an undisclosed fee.",
 "The UK economy grew by 0.3% in the first quarter of 2024, according to figures from the Office for National Statistics.",
 "Residents of a village in Cornwall say they were left without water for three days after a burst pipe on Monday.",
 "Former Labour MP David Harlow has been appointed chair of the new housing commission, the prime minister has said.",
 "Two climbers have died after being caught in an avalanche on Ben Nevis, Lochaber Mountain Rescue Team has confirmed.",
]
PLAIN = [
 "The committee will meet again on Thursday to discuss the proposal for the new road.",
 "I would like to thank the President and the Commission for their work on this report.",
 "We must ensure that every citizen has access to clean water and reliable energy.",
 "The weather was cold and rainy, so we decided to stay at home and read a book.",
 "Mr President, ladies and gentlemen, the debate on fisheries policy is closed.",
 "The new regulation will enter into force on 1 January and applies to all member states.",
 "Prices for fresh vegetables have risen sharply in recent months, which worries many families.",
 "She explained that the company had decided to move its headquarters to another city next year.",
 "In my opinion, the proposed amendment does not address the real problem facing small businesses.",
 "The train from Berlin arrived late because of heavy snow, and many passengers missed their connections.",
 "The two sides agreed to continue the negotiations, although serious differences remain on trade and security.",
 "He has lived in the same small house near the river for more than thirty years.",
]
TPL = [
 "{a} went to {p} with {b}. They had {n} apples and ate them on {d}.",
 "One day, {a} found {n} coins near {p}. {a} gave them to {b}.",
 "{b} was sad. {a} said, \"Let us go to {p} on {d}.\" They were happy.",
 "Once upon a time, {a} and {b} lived in {p}. {a} had {n} cats.",
]
RAW = dict(a=["Okonkwo", "Mlauzi", "Yordanka", "Dragomir", "Siobhan", "Jürgen Müller", "Zdravko Petkov", "Anneliese Schröder"],
           b=["Hristo", "Ishan", "Ngozi", "Wojciech", "Aurelio", "Günther Weiß", "Stoyan Georgiev", "Björn Köhler"],
           p=["Plovdiv", "Veliko Tarnovo", "Ouagadougou", "Szczecin", "Reykjavik", "Gütersloh", "Stara Zagora", "Freiburg im Breisgau"],
           n=["340", "17", "2026", "4.5", "981"],
           d=["3.11.2026", "Tuesday 14 May", "12/07", "2026-10-30", "the 9th of June"])
SUR = dict(a=["Tom"], b=["Lily"], p=["the park"], n=["three"], d=["Monday"])
TAG = dict(a=["<P1>"], b=["<P2>"], p=["<L1>"], n=["<N1>"], d=["<D1>"])
