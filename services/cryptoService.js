import axios from "axios";
import moment from "moment";

const formatSparkline = (numbers) => {
  const sevenDaysAgo = moment().subtract(7, "days").unix();
  let formattedSparkline = numbers.map((item, index) => {
    return {
      timestamp: sevenDaysAgo + (index + 1) * 3600,
      value: item,
    };
  });
  return formattedSparkline;
};

const formatMatketData = (data) => {
    let formattedData = [];
    data.forEach(item => {
        const formattedSparkline =formatSparkline(item.sparkline_in_7d.price)
        const formattedItem = {
            ...item,
            sparkline_in_7d:{
                price:formattedSparkline
            }
        }
        formattedData.push(formattedItem);
    });
return formattedData;
}
export const getMarketData = async () => {
  try {
    const response = await axios.get(
      "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=20&page=1&sparkline=true&price_change_percentage=7d&locale=en"
    );
    const data = response.data;
    const formattedResponse = formatMatketData(data);
    return formattedResponse;
  } catch (error) {
    console.log(error.message);
  }
};
