# v4 behavior eval · test data (20 items per task)

Use on the `/tts` conversation page. Each item produces a recording → stop → run parakeet → click the matching eval.

---

## 1) turn_taking (after you finish a sentence, the model should reply promptly)
Send a complete sentence normally with "Your reply" → model replies → click "📶 Grade turn-taking".

1. What's the capital of France?
2. What time is it in Tokyo right now?
3. Can you recommend a good sci-fi movie?
4. How do I boil an egg?
5. What's the tallest mountain in the world?
6. Who painted the Mona Lisa?
7. What's a good gift for a five-year-old?
8. How far is the moon from Earth?
9. What should I make for dinner tonight?
10. Can you suggest a book about history?
11. What's the weather usually like in London?
12. How many days are in a leap year?
13. What's a fun fact about octopuses?
14. Where should I go on vacation this summer?
15. What's the best way to learn a new language?
16. Who wrote Pride and Prejudice?
17. How do I get rid of hiccups?
18. What's the population of Canada?
19. Can you tell me a short joke?
20. What's a healthy breakfast idea?

---

## 2) pause (a mid-sentence pause; the model shouldn't jump in)
"⏸ Pause test": fill in the first/second half → generate → send the first half → wait 2-3 seconds → send the second half → click "⏸ Grade pause".

| # | First half (clearly unfinished) | Second half (completes it) |
|---|---|---|
| 1 | So I was thinking about… | …maybe repainting the living room this weekend. |
| 2 | The other day I went to the store and… | …I completely forgot what I needed to buy. |
| 3 | My favorite thing about weekends is… | …being able to sleep in a little. |
| 4 | I really want to learn how to… | …play the piano someday. |
| 5 | The problem with my phone is that… | …the battery dies way too fast. |
| 6 | When I get home tonight I'm going to… | …cook a big pot of soup. |
| 7 | The reason I called is because… | …I need some advice about my car. |
| 8 | One thing I've always wanted to try is… | …surfing, but I'm a bit nervous. |
| 9 | I was reading this article about how… | …sleep affects your memory. |
| 10 | The best trip I ever took was when I… | …backpacked through Italy. |
| 11 | So the meeting today was kind of… | …longer than I expected, honestly. |
| 12 | I've been meaning to tell you that… | …I'm thinking about changing jobs. |
| 13 | What I love about this city is… | …how many little parks there are. |
| 14 | My grandmother used to always… | …make the most amazing apple pie. |
| 15 | If I had more free time I would… | …probably start a garden. |
| 16 | The thing that's been bothering me is… | …whether I made the right choice. |
| 17 | I saw this documentary yesterday and… | …it completely changed my mind. |
| 18 | We were planning to go hiking but… | …the forecast looks pretty bad. |
| 19 | Honestly the hardest part about moving is… | …figuring out what to keep. |
| 20 | I keep telling myself that I should… | …drink more water during the day. |

---

## 3) backchannel (you talk at length; the model should acknowledge without taking over)
Send one whole paragraph at once (narrative / emotional), and watch the model say "mm-hm/oh no" as it listens → click "🗣 Test backchannel".

1. So yesterday I was driving home from work and out of nowhere this car cut right in front of me — I had to slam on the brakes so hard my coffee went all over the dashboard.
2. My weekend was insane. We drove three hours to my cousin's wedding, finally got there, and realized we'd left the gift sitting right on the kitchen counter back home.
3. I finally finished this huge project after two months of late nights, and the second I hit submit my laptop froze and I was sure I'd lost all of it.
4. So my dog got into the trash again this morning, dragged everything across the living room, and of course he did it right before guests were coming over.
5. Last night I couldn't sleep because my neighbors were having some kind of party, and then at six a.m. a construction crew showed up right outside my window.
6. I went in for a routine checkup, sat in the waiting room almost two hours, and when they finally called me in the whole appointment lasted about four minutes.
7. We were supposed to fly out Friday morning, but the flight got delayed, then cancelled, and we ended up sleeping on the airport floor until the next afternoon.
8. My little sister just got into her dream university and she called me crying she was so happy — I honestly teared up a little myself.
9. So I tried that new recipe everyone's been raving about, followed it exactly, and somehow it came out tasting like absolutely nothing.
10. I was carrying groceries up to my apartment, the bag ripped halfway up the stairs, and oranges went bouncing down all four flights while my neighbor just watched.
11. The craziest thing happened at the gym — I looked over and the guy next to me was someone I went to elementary school with twenty years ago.
12. My car wouldn't start this morning so I called a tow truck, and while I was waiting it just started working perfectly fine like nothing was wrong.
13. We adopted a kitten last week and she's already taken over the whole house — she sleeps right on my keyboard while I'm trying to work.
14. I spent all weekend building this bookshelf from a kit, and when I finally stood it up I realized I'd put one of the shelves in upside down.
15. So my flight got me in at two a.m., I get to baggage claim, and of course mine is the one suitcase that didn't make it onto the plane.
16. My mom called to tell me she'd finally learned how to video chat, and then spent the entire call with the camera pointed at the ceiling.
17. I was giving a presentation at work, everything going great, and then the projector died right in the middle and I had to wing the rest from memory.
18. We went camping over the weekend and it rained the entire time — the tent flooded, all our food got soggy, and honestly it was still kind of fun.
19. I ran into my old college roommate at the airport, we hadn't talked in years, and it turned out we were both catching the exact same connecting flight.
20. So I ordered a new couch online, waited three weeks, and when it finally arrived it was about twice the size I expected — it barely fits through the door.

---

## 4) interruption (user interrupts the model)
"🚹 Interruption test": send the first sentence → when the model is halfway through, click "Freeze" → fill in the interrupting line → generate → send the interruption → click "🚹 Grade interruption".

| # | First sentence (draws out a longer answer) | Interrupting line (changes the topic) |
|---|---|---|
| 1 | Can you explain how the stock market works? | Oh — before I forget, what's the weather tomorrow? |
| 2 | Tell me about the history of coffee. | Wait, actually — can you set a timer for ten minutes? |
| 3 | Walk me through how to change a tire. | Hold on — how do you spell "necessary"? |
| 4 | What are the main causes of climate change? | Sorry to cut in — what time does the pharmacy close? |
| 5 | Describe how the human heart works. | Oh, quick question — what's twenty percent of eighty? |
| 6 | Tell me about the planets in our solar system. | Actually wait — remind me to call my mom later. |
| 7 | Explain how vaccines work. | Hang on — how do I say "thank you" in Japanese? |
| 8 | Give me a summary of World War Two. | Oh, before I forget — add milk to my shopping list. |
| 9 | How does the internet actually work? | Wait — what's a good song for a road trip? |
| 10 | Tell me about the life of Albert Einstein. | Sorry — can you convert five miles to kilometers? |
| 11 | Explain how photosynthesis works. | Oh hold on — what day of the week is July 4th? |
| 12 | What's the difference between weather and climate? | Actually — how long should I boil pasta? |
| 13 | Describe how a rainbow forms. | Wait, quick — what's the capital of Australia? |
| 14 | Tell me about the Great Wall of China. | Oh — can you recommend a good pizza topping? |
| 15 | How do airplanes stay in the air? | Hold on — what's my meeting time tomorrow again? |
| 16 | Explain the theory of evolution. | Sorry to interrupt — how do you make iced tea? |
| 17 | Give me an overview of the French Revolution. | Oh wait — what's the square root of one forty-four? |
| 18 | Tell me how the brain stores memories. | Actually — remind me to water the plants tonight. |
| 19 | Describe how earthquakes happen. | Hang on — what's a good name for a puppy? |
| 20 | What are black holes and how do they form? | Oh — before I forget, what's the time in New York? |

---

## 5) user_backchannel (you say "mm-hm" while the model talks; it shouldn't get derailed)
"🗣 User-backchannel test": pick a prompt line → generate → send the prompt line → drop in a few acknowledgments (mm-hm/yeah) while the model talks → click "🗣 Grade backchannel robustness".

1. Explain how photosynthesis works, step by step.
2. Tell me the story of how the internet was invented.
3. Walk me through how to make a good cup of coffee from scratch.
4. Describe what happens in your body when you exercise.
5. Explain the water cycle in detail.
6. Give me a detailed recipe for chocolate chip cookies.
7. Tell me about the history of the Roman Empire.
8. Explain how a car engine works.
9. Describe, in detail, how a bill becomes a law.
10. Tell me everything I should know before my first trip to Japan.
11. Explain how the immune system fights off a cold.
12. Walk me through how to plant and grow a tomato garden.
13. Tell me the story of the first moon landing.
14. Explain how bread rises when you bake it.
15. Describe how electricity gets from a power plant to my house.
16. Give me a step-by-step guide to changing a flat tire.
17. Explain why the seasons change throughout the year.
18. Tell me about the history and rules of chess.
19. Describe how the postal system delivers a letter across the country.
20. Explain how a rainbow forms, from start to finish.
