The scheduled time has arrived. Execute this scheduled cron job now and report the result to the user in the same session.

Rules:
- Speak directly to the user in their language.
- Do not narrate internal progress.
- Do not include user IDs.
- Do not add status reports like "Done" or "Reminded" unless they are the natural response.
- If the job says to act only under a condition and nothing warrants a message this run, reply with exactly NO_MESSAGE — nothing will be delivered to the user.

Cron job: {{ message }}
