下面是一个 QQ 群的一段聊天记录，{name}是群里的一员（她的发言标着「(她自己)」）。这段聊天已经过去一会儿了，你能看到每句话之后大家是怎么接的。请事后判断两件事。

聊天记录（行首是消息编号；「→某人」表示用 @ 或引用明确指向了谁，「→她」是指向{name}；「[回复 #编号 某人: …]」是引用的那句话）：

{lines}

一、这些消息当时是在对谁说：{targets}

- her：在对{name}说，等她回应（叫她、问她、和她接着聊她刚说的事）。
- other：在对另一个（或另几个）群友说。
- group：对群里所有人说，没有特定对象（提问、分享、感叹）。
- self：自言自语，或者只是个反应（"草""哈哈"、表情包），不需要谁接。紧跟在{name}后面笑一下、感叹一句，也算 self，不算 her。
- unclear：看完前后文也说不准。

要看后面的发展：是谁接了这句话、发言人后来有没有说"我是问你""没问你"、话题是在谁和谁之间继续的。「你」指谁，要看这句话在接谁的话。只有明显是对{name}说的才选 her。

二、{name}的这些发言：{own}

每条后面标着她当时回应的那句话（不知道的写着「不知道」）。判断：

- for_her：她回应的那句话当时是不是对她说的：yes / no / unclear。
- answered_by_other：那句话本来要问的人后来有没有自己回答：yes / no / unclear（for_her 是 yes 时填 unclear）。
- stance：大家对她这句话的反应：appreciation（认可、感谢、夸她）/ playful（接梗、开玩笑地接）/ neutral（正常接话）/ ignored（没人理）/ confusion（没看懂、问她在说什么）/ correction（纠正她说错的事）/ rejection（"没问你""不是跟你说的"、嫌她插嘴）/ attack（骂她）。
- contribution：她这句话对聊天的作用：advance（推进了话题）/ maintain（自然地接住了）/ acknowledge（只是应了一声）/ close（让话题收尾了）/ wrong_push（答非所问，或者误会了在问谁）/ intrusion（打断了别人之间的对话）。

只根据记录里看得到的反应判断，不要猜；没人接就是 ignored。

调用 record_hindsight 提交，每条都要填：lines 里每项是 {"row": "#编号", "to": "her|other|group|self|unclear"}；replies 里每项是 {"row": "#编号", "for_her": …, "answered_by_other": …, "stance": …, "contribution": …}。她没有发言时 replies 为空。
