import axios from 'axios';

async function fetchData() {
    try {
        const response = await axios.get('http://127.0.0.1:5000/api/data');
        console.log('Data from API:', response.data);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

fetchData();
