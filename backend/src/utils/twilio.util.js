import twilio from "twilio";

const twilioClient = twilio(
    process.env.TWILIO_CLIENT_ACCOUNT_SID,
    process.env.TWILIO_CLIENT_AUTH_TOKEN
);

export async function sendMessage(to, body) {
    return message = await twilioClient.messages.create({
        body,
        from: process.env.TWILIO_CLIENT_PHONE_NUMBER,
        to,
    });
}

export default twilioClient;