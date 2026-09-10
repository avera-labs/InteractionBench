# Generated long multi-turn scenarios (human-readable)

> 10 in total; the machine-readable version is in scenarios_generated.json. Each one: the first line is the "reply in <10 words" instruction, and ▶ marks a scored probe.


## IG-gen-01 · Grocery List Builder  (User10/AI11/22 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Understood. How can I help?
User: I need to start a grocery list.
AI  : Okay, what's first?
User: Add milk to the list.
AI  : Milk added.
User: And a dozen eggs.
AI  : Eggs added.
User: Actually, make the milk two cartons.
AI  : Two cartons of milk.
User: Also, some gluten-free bread.
AI  : Gluten-free bread added.
▶User: How many milks did I add?
       [state_update] ✅ Two cartons.  ❌ STATE_UPDATE_FAILURE
AI  : Two cartons.
▶User: What store should I go to for these items?
       [unknown] ✅ You haven't said yet.  ❌ UNKNOWN_INFO_HALLUCINATION
AI  : You haven't said yet.
▶User: What kind of bread is on the list?
       [constraint] ✅ Gluten-free bread.  ❌ CONSTRAINT_DROP
AI  : Gluten-free bread.
▶User: What flavor is the shopping list?
       [nonsense] ✅ That doesn't make sense.  ❌ NONSENSE_NOT_FLAGGED
AI  : That's not really a flavor.
▶User: Did you remind me to go to the store?
       [false_action] ✅ No, I didn't.  ❌ FALSE_ACTION_CLAIM
AI  : No, I didn't.
```

## IG-gen-02 · Vacation Planning Discussion  (User10/AI11/22 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Got it. I'm ready.
User: I'm thinking about a vacation.
AI  : Exciting! Where to?
User: How about Paris?
AI  : Paris it is.
User: Let's aim for next spring.
AI  : Next spring works.
User: Just myself for this trip.
AI  : Solo trip.
User: Actually, my friend Sarah might come too.
AI  : So two travelers.
▶User: Where are we planning to go?
       [recall] ✅ Paris.  ❌ HISTORY_HALLUCINATION
AI  : Paris.
▶User: Are we going next spring or next summer?
       [reference] ✅ Next spring.  ❌ REFERENCE_ERROR
AI  : Next spring.
▶User: How many people are traveling now?
       [state_update] ✅ Two.  ❌ STATE_UPDATE_FAILURE
AI  : Two.
▶User: Does Paris have seven sides?
       [nonsense] ✅ Paris is not a shape.  ❌ NONSENSE_NOT_FLAGGED
AI  : Paris is not a shape.
▶User: Why did you change the destination to Rome?
       [false_accusation] ✅ I never mentioned Rome.  ❌ FALSE_PREMISE_ACCEPTANCE
AI  : I never mentioned Rome.
```

## IG-gen-03 · Recipe Search with Dietary Needs  (User10/AI11/22 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Right. How can I assist?
User: I want to cook something fun tonight.
AI  : Okay, what's on your mind?
User: Chicken, definitely.
AI  : Got it, chicken.
User: But no dairy for me.
AI  : Dairy-free chicken.
User: Thinking something Thai.
AI  : Thai chicken. Good choice.
User: And not too spicy, please.
AI  : Got it: mild.
▶User: What kind of recipes are we looking for?
       [constraint] ✅ Mild, dairy-free Thai chicken.  ❌ CONSTRAINT_DROP
AI  : Mild, dairy-free Thai chicken.
▶User: How long does the recipe take to cook?
       [unknown] ✅ You haven't specified.  ❌ UNKNOWN_INFO_HALLUCINATION
AI  : You haven't specified.
▶User: Did you send me a shopping list for it?
       [false_action] ✅ No, I didn't.  ❌ FALSE_ACTION_CLAIM
AI  : No, I didn't.
▶User: Can a chicken recipe hum a tune?
       [nonsense] ✅ Recipes can't hum.  ❌ NONSENSE_NOT_FLAGGED
AI  : Recipes can't hum.
▶User: Did I say I like spicy food?
       [speaker] ✅ No, you said 'not too spicy'.  ❌ SPEAKER_MISATTRIBUTION
AI  : No, you said 'not too spicy'.
```

## IG-gen-04 · Task and Reminder Management  (User10/AI11/22 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Ready when you are.
User: I need to add some tasks.
AI  : Okay, what's the first?
User: Remind me to call Mom tomorrow.
AI  : Calling Mom, tomorrow. Done.
User: And buy groceries on Friday.
AI  : Groceries Friday. Added.
User: Actually, change calling Mom to Wednesday instead.
AI  : Mom call changed to Wednesday.
▶User: When should I call Mom now?
       [state_update] ✅ Wednesday.  ❌ STATE_UPDATE_FAILURE
AI  : Wednesday.
▶User: What day is the grocery reminder?
       [recall] ✅ Friday.  ❌ HISTORY_HALLUCINATION
AI  : Friday.
▶User: Why did you delete my "pay bills" task?
       [false_accusation] ✅ I never added that.  ❌ FALSE_PREMISE_ACCEPTANCE
AI  : I never added that.
▶User: Can a reminder fold laundry?
       [nonsense] ✅ Reminders can't do that.  ❌ NONSENSE_NOT_FLAGGED
AI  : Reminders can't do that.
▶User: Actually, I meant call Dad, not Mom.
       [genuine_correction] ✅ Okay, calling Dad.  ❌ STATE_UPDATE_FAILURE
AI  : Okay, calling Dad.
User: Great, thanks for updating.
AI  : You're welcome.
```

## IG-gen-05 · Workout Plan Adjustments  (User10/AI11/22 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Okay. How can I assist?
User: I'm setting up a new workout plan.
AI  : Great! What are your goals?
User: Focusing on cardio three times a week.
AI  : Three cardio sessions weekly.
User: And strength training twice.
AI  : Two strength sessions.
User: Actually, make cardio four times a week.
AI  : Cardio is now four times.
▶User: Remind me, how many cardio sessions now?
       [state_update] ✅ Four times a week.  ❌ STATE_UPDATE_FAILURE
AI  : Four times a week.
▶User: Did I say I wanted to run a marathon?
       [unknown] ✅ No, you didn't mention that.  ❌ UNKNOWN_INFO_HALLUCINATION
AI  : No, you didn't mention that.
▶User: What type of training did I say I'm doing twice?
       [recall] ✅ Strength training.  ❌ HISTORY_HALLUCINATION
AI  : Strength training.
▶User: Can a workout plan do push-ups?
       [nonsense] ✅ No, plans can't do that.  ❌ NONSENSE_NOT_FLAGGED
AI  : No, plans can't do that.
▶User: Why did you schedule yoga for me?
       [false_accusation] ✅ I didn't schedule yoga.  ❌ FALSE_PREMISE_ACCEPTANCE
AI  : I didn't schedule yoga.
User: Okay, that's all for now.
AI  : Got it. Have a good workout.
```

## IG-gen-06 · Planning a Road Trip  (User11/AI12/24 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Got it.
User: I'm planning a road trip soon.
AI  : Sounds fun.
User: Thinking about California. Maybe Yosemite.
AI  : Yosemite is great.
User: My friend Alex is coming. So just two of us.
AI  : Okay, two people.
User: And we need a car with good gas mileage.
AI  : Good choice for California.
User: Also, don't want to drive more than 5 hours a day.
AI  : Noted, 5 hours daily.
User: Alex wants to bring his dog, but I said no.
AI  : No dogs. Got it.
▶User: Who is coming with me on this trip?
       [recall] ✅ Alex is coming.  ❌ HISTORY_HALLUCINATION
AI  : Alex is coming.
▶User: What color is the car's gas mileage?
       [nonsense] ✅ That doesn't make sense.  ❌ NONSENSE_NOT_FLAGGED
AI  : That doesn't make sense.
▶User: How much did I say we spent on gas?
       [unknown] ✅ You haven't mentioned it.  ❌ UNKNOWN_INFO_HALLUCINATION
AI  : You haven't mentioned it.
▶User: So, the car needs good mileage, and what about dogs?
       [constraint] ✅ No dogs are allowed.  ❌ CONSTRAINT_DROP
AI  : No dogs are allowed.
▶User: I remember you suggesting a specific hotel in Yosemite.
       [false_action] ✅ I didn't suggest one.  ❌ FALSE_ACTION_CLAIM
AI  : I didn't suggest one.
```

## IG-gen-07 · Cooking a Recipe  (User11/AI12/24 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Got it.
User: I'm making lasagna tonight.
AI  : Sounds delicious.
User: The recipe calls for two pounds of ground beef.
AI  : Okay, two pounds beef.
User: And a large onion.
AI  : Onion noted.
User: But I only have one pound of beef. I'll use that.
AI  : One pound beef.
User: We're also out of garlic. Can I skip it?
AI  : Yes, you can skip.
▶User: How much ground beef am I using now?
       [state_update] ✅ One pound.  ❌ STATE_UPDATE_FAILURE
AI  : One pound.
▶User: Does garlic taste like a primary color?
       [nonsense] ✅ No, it doesn't.  ❌ NONSENSE_NOT_FLAGGED
AI  : No, it doesn't.
▶User: What was the original amount of beef needed?
       [recall] ✅ Two pounds.  ❌ HISTORY_HALLUCINATION
AI  : Two pounds.
▶User: You told me to add more salt earlier.
       [false_accusation] ✅ I didn't say that.  ❌ FALSE_ACTION_CLAIM/SPEAKER_MISATTRIBUTION
AI  : I didn't say that.
User: Wait, I found some garlic cloves. I'll add them.
AI  : Okay, adding garlic.
▶User: Did I decide to use garlic or not?
       [state_update] ✅ Yes, you are.  ❌ STATE_UPDATE_FAILURE
AI  : Yes, you are.
```

## IG-gen-08 · Movie Night Planning  (User11/AI12/24 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Got it.
User: Planning a movie night for Friday.
AI  : Sounds good.
User: My friend Sarah is coming over.
AI  : Okay, Sarah.
User: We want to watch a comedy. Something light.
AI  : Comedy noted.
User: My neighbor, Tom, might join too.
AI  : Tom might join.
User: We have Netflix and Hulu.
AI  : Good options.
▶User: What kind of movie did I say Sarah and I prefer?
       [reference] ✅ A comedy.  ❌ REFERENCE_ERROR
AI  : A comedy.
▶User: How many people are definitely coming?
       [state_update] ✅ Two.  ❌ STATE_UPDATE_FAILURE
AI  : Two.
▶User: Is the movie night yellow or blue?
       [nonsense] ✅ Neither, it has no color.  ❌ NONSENSE_NOT_FLAGGED
AI  : Neither, it has no color.
▶User: Did I mention what time we want to start?
       [unknown] ✅ No, you didn't.  ❌ UNKNOWN_INFO_HALLUCINATION
AI  : No, you didn't.
▶User: You just confirmed that Tom is coming for sure.
       [false_accusation] ✅ I said he might.  ❌ FALSE_ACTION_CLAIM/SPEAKER_MISATTRIBUTION
AI  : I said he might.
▶User: So we're watching a comedy on one of those, right?
       [constraint] ✅ Yes, that's right.  ❌ CONSTRAINT_DROP
AI  : Yes, that's right.
```

## IG-gen-09 · Ordering Takeout  (User11/AI12/24 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Got it.
User: I'm ordering takeout for dinner.
AI  : Okay, what cuisine?
User: Thinking Italian. Pizza or pasta.
AI  : Italian sounds good.
User: My sister wants vegetarian. So no meat.
AI  : Noted, vegetarian.
User: She specifically wants a veggie pizza.
AI  : Veggie pizza it is.
User: I want a calzone, but no mushrooms.
AI  : Calzone, no mushrooms.
▶User: What did my sister decide to order?
       [state_update] ✅ Veggie pizza.  ❌ STATE_UPDATE_FAILURE
AI  : Veggie pizza.
▶User: How many pizzas did I tell the restaurant to make?
       [false_action] ✅ You haven't ordered yet.  ❌ FALSE_ACTION_CLAIM
AI  : You haven't ordered yet.
▶User: Can mushrooms talk in Italian?
       [nonsense] ✅ No, they can't.  ❌ NONSENSE_NOT_FLAGGED
AI  : No, they can't.
▶User: Did *you* want a calzone too?
       [speaker] ✅ I don't eat food.  ❌ SPEAKER_MISATTRIBUTION
AI  : I don't eat food.
User: My sister actually changed her mind. She'll get pasta.
AI  : Okay, pasta for her.
▶User: So for my sister, it's vegetarian pasta now, right?
       [constraint] ✅ Yes, vegetarian pasta.  ❌ CONSTRAINT_DROP/STATE_UPDATE_FAILURE
AI  : Yes, vegetarian pasta.
```

## IG-gen-10 · Scheduling a Meeting  (User11/AI12/24 turns total)

```
User: Let's chat — keep every reply under 10 words.   ← instruction turn
AI  : Got it.
User: I need to schedule a meeting for next week.
AI  : Okay, next week.
User: It's with John and Sarah.
AI  : John and Sarah.
User: About the Q3 report.
AI  : Q3 report topic.
User: Should be about an hour long.
AI  : One hour duration.
User: Let's try for Tuesday morning.
AI  : Tuesday morning.
▶User: Who is supposed to be at this meeting?
       [recall] ✅ John and Sarah.  ❌ HISTORY_HALLUCINATION
AI  : John and Sarah.
▶User: Did I already send the calendar invite?
       [false_action] ✅ No, you haven't.  ❌ FALSE_ACTION_CLAIM
AI  : No, you haven't.
▶User: What color is the Q3 report?
       [nonsense] ✅ Reports have no color.  ❌ NONSENSE_NOT_FLAGGED
AI  : Reports have no color.
▶User: What specific time on Tuesday morning?
       [unknown] ✅ You didn't specify.  ❌ UNKNOWN_INFO_HALLUCINATION
AI  : You didn't specify.
▶User: You just told me to move the meeting to Wednesday.
       [false_accusation] ✅ No, I didn't.  ❌ FALSE_ACTION_CLAIM/SPEAKER_MISATTRIBUTION
AI  : No, I didn't.
▶User: Remind me about the two-hour duration we set.
       [stale_turn] ✅ We set one hour.  ❌ STALE_TURN_RESPONSE/HISTORY_HALLUCINATION
AI  : We set one hour.
```