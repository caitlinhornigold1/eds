const chatLauncher =
document.getElementById("chatLauncher");

const chatPanel =
document.getElementById("chatPanel");

const closeChatButton =
document.getElementById("closeChat");

const questionInput =
document.getElementById("question");


function openChat() {
chatPanel.hidden = false;

chatLauncher.setAttribute(
    "aria-expanded",
    "true"
);

questionInput.focus();
}


function closeChat() {
chatPanel.hidden = true;

chatLauncher.setAttribute(
    "aria-expanded",
    "false"
);

chatLauncher.focus();
}


chatLauncher.addEventListener(
"click",
function() {
    if (chatPanel.hidden) {
        openChat();
    } else {
        closeChat();
    }
}
);


closeChatButton.addEventListener(
"click",
closeChat
);


document.addEventListener(
"keydown",
function(event) {
    if (
        event.key === "Escape" &&
        !chatPanel.hidden
    ) {
        closeChat();
    }
}
);

function formatAIResponse(text) {
    if (!text) return "";

    return text
        // 1. Fix isolated colons on newlines coming from the LLM (e.g., "**Term**\n:")
        .replace(/(\*\*[^*]+\*\*)\s*\n+\s*:\s*/g, "$1: ")

        // 2. Convert consecutive * bullet points into <ul><li>...</li></ul>
        .replace(/(?:^|\n)((?:\* .+(?:\n|$))+)/g, function(match, list) {
            const items = list
                .trim()
                .split("\n")
                .map(function(item) {
                    return "<li>" + item.replace(/^\* /, "") + "</li>";
                })
                .join("");

            return '<ul class="response-list">' + items + "</ul>";
        })

        // 3. Convert **bold text** into strong tags
        .replace(/\*\*(.*?)\*\*/g, '<strong class="response-heading">$1</strong>')

        // 4. Convert remaining paragraph line breaks into <br> (ignoring newlines inside HTML tags)
        .replace(/\n\n+/g, "<br><br>")
        .replace(/(?<!<\/li>|<\/ul>)\n/g, "<br>");
}

async function askQuestion() {
const question =
    questionInput.value.trim();

const conversation =
    document.getElementById(
        "conversation"
    );

const sendButton =
    document.getElementById(
        "sendButton"
    );

if (question === "") {
    return;
}


sendButton.disabled = true;


/* User message */

const userMessage =
    document.createElement("p");

    userMessage.classList.add(
        "user-message"
    );

const userLabel =
    document.createElement("strong");

userMessage.appendChild(userLabel);

userMessage.appendChild(
    document.createTextNode(question)
);

conversation.appendChild(
    userMessage
);


questionInput.value = "";


/* AI loading message */

const aiMessage =
    document.createElement("p");

const aiLabel =
    document.createElement("strong");

    aiMessage.classList.add(
        "eds-message"
    );

const aiText =
    document.createElement("span");

    aiText.classList.add("ai-response");

aiText.textContent = "Thinking...";

aiMessage.appendChild(aiLabel);
aiMessage.appendChild(aiText);

conversation.appendChild(
    aiMessage
);


conversation.scrollTop =
    conversation.scrollHeight;


try {
    const response =
        await fetch(
            "http://127.0.0.1:8000/chat",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        question: question
                    })
            }
        );


    if (!response.ok) {
        throw new Error(
            "Server returned an error"
        );
    }


    const data =
        await response.json();


    /* Unanswered question */

    if (data.unanswered) {
        aiText.textContent =
            "I couldn't find relevant information in the EDS knowledge base to answer this question.";

        aiMessage.style.backgroundColor =
            "#f8d7da";

        aiMessage.style.border =
            "1px solid #f1aeb5";

        aiMessage.style.padding =
            "10px";

        aiMessage.style.fontWeight =
            "bold";

    } else {
        aiText.innerHTML =
    formatAIResponse(data.answer);
    }


    /* Low-confidence warning */

    if (
        data.low_confidence &&
        !data.unanswered
    ) {
        const warning =
            document.createElement("p");

        warning.setAttribute(
            "role",
            "status"
        );

        warning.textContent =
            "Warning: This answer may not be fully supported by the available EDS information.";

        warning.style.backgroundColor =
            "#fff3cd";

        warning.style.border =
            "1px solid #ffe69c";

        warning.style.padding =
            "10px";

        warning.style.fontWeight =
            "bold";

        conversation.appendChild(
            warning
        );
    }


      /* Sources */

      if (
        data.sources &&
        data.sources.length > 0
    ) {
        const sourcesContainer =
            document.createElement("div");

        sourcesContainer.classList.add(
            "sources"
        );

        const sourcesList =
            document.createElement("ul");


        data.sources.forEach(
            function(source) {

                const sourceItem =
                    document.createElement("li");

                sourceItem.classList.add(
                    "source-item"
                );


                const sourceLink =
                    document.createElement("a");

                sourceLink.href =
                    source.url;

                sourceLink.target =
                    "_blank";

                sourceLink.rel =
                    "noopener noreferrer";


                let sourceText =
                    source.document;


                if (
                    source.page !== undefined &&
                    source.page !== null
                ) {
                    sourceText +=
                        " - Page " +
                        source.page;
                }


                sourceLink.textContent =
                    sourceText;


                sourceLink.setAttribute(
                    "aria-label",
                    "Open source: " +
                    source.document
                );


                sourceItem.appendChild(
                    sourceLink
                );

                sourcesList.appendChild(
                    sourceItem
                );
            }
        );


        sourcesContainer.appendChild(
            sourcesList
        );

        conversation.appendChild(
            sourcesContainer
        );
    }


} catch (error) {
    aiText.textContent =
        "Sorry, something went wrong while processing your question. Please try again.";

    console.error(error);


} finally {
    sendButton.disabled =
        false;

    questionInput.focus();

    conversation.scrollTop =
        conversation.scrollHeight;
}
}


document
.getElementById("chatForm")
.addEventListener(
    "submit",
    function(event) {
        event.preventDefault();

        askQuestion();
    }
);