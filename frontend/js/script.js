// getElementById('predictionForm'): obtiene el formulario por su id, addEventListener('submit'): ejecuta código al enviar el formulario, async function(event): función asíncrona que recibe el evento del submit del html
document.getElementById('btnCalcular').addEventListener('click', async function () {

    // 2. Capturamos los valores de las cajas de texto del HTML
    const kmsValue = parseFloat(document.getElementById('kms').value);
    const yearValue = parseInt(document.getElementById('year').value);
    const premiumValue = parseFloat(document.getElementById('premium').value);
    if (isNaN(kmsValue) || isNaN(yearValue) || isNaN(premiumValue)) {
        alert("Por favor, rellena todos los campos con valores numéricos válidos.");
        return;
    }

    // 3. Estructuramos el Payload en formato JSON
    const payload = {
        "OBJECT_ID": 1,
        "INSR_BEGIN": "2024-01-01",
        "INSR_END": "2025-01-01",
        "DURATION": 366,
        "INSR_TYPE": "1202",
        "PROD_YEAR": yearValue,          // <-- Coge el valor dinámico del HTML
        "PREMIUM": premiumValue,         // <-- Coge el valor dinámico del HTML
        "INSURED_VALUE": 35000,
        "SEX": "1",
        "MAKE": "TOYOTA",
        "USAGE": "Private",
        "TYPE_VEHICLE": "Car",
        "SEATS_NUM": 5,
        "CARRYING_CAPACITY": 0,
        "CCM_TON": kmsValue,             // <-- Coge el valor dinámico del HTML
        "EFFECTIVE_YR": 2024
    };
    try {
        // 4. Enviamos la petición HTTP POST asíncrona hacia la API de FastAPI
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            console.log("¡ÉXITO! La API ha respondido bien. Datos recibidos:", data); // luego se puede quitar
            // Leemos la clave exacta que devuelve tu app.py
            const resultAmount = data.predicted_claim_amount;

            // Actualizamos el texto en el HTML
            document.getElementById('resultValue').innerText = `${parseFloat(resultAmount).toFixed(2)}€`;

            const block = document.getElementById('resultBlock');
            block.className = "mt-6 p-4 bg-green-50 border border-green-200 rounded-xl text-center transition-colors duration-300";
            document.getElementById('resultValue').className = "text-3xl font-extrabold text-green-600 mt-1";
        } else {
            console.log("¡ERROR! La API respondió pero con un código de fallo (ej. 422 o 500).");//luego se puede quitar
            alert('Error en la respuesta del servidor de predicción.');
        }
    } catch (error) {
        console.error('Error en la conexión:', error);
        alert('No se pudo conectar con la API de SecureCarX. Asegúrate de que el contenedor esté corriendo.');
    }
});