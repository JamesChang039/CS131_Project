import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAYmHpgkDg-nQAMGE4dQzM22yuyeJnnLxM",
  authDomain: "cs131-final-project-497022.firebaseapp.com",
  projectId: "cs131-final-project-497022",
  storageBucket: "cs131-final-project-497022.firebasestorage.app",
  messagingSenderId: "828167211823",
  appId: "1:828167211823:web:f6dd593f138a68691c6fda",
  measurementId: "G-ZGGV2WRGG3"
};

const app = initializeApp(firebaseConfig);

export const db = getFirestore(app);