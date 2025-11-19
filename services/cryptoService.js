import axios from "axios";
import moment from "moment";

const formatSparkline = (numbers) => {
  if (!numbers || !Array.isArray(numbers)) {
    return [];
  }
  const sevenDaysAgo = moment().subtract(7, "days").unix();
  let formattedSparkline = numbers.map((item, index) => {
    return {
      timestamp: sevenDaysAgo + (index + 1) * 3600,
      value: item,
    };
  });
  return formattedSparkline;
};

const formatMarketData = (data) => {
    if (!data || !Array.isArray(data)) {
      return [];
    }
    let formattedData = [];
    data.forEach(item => {
        if (item && item.sparkline_in_7d && item.sparkline_in_7d.price) {
          const formattedSparkline = formatSparkline(item.sparkline_in_7d.price);
          const formattedItem = {
              ...item,
              sparkline_in_7d:{
                  price:formattedSparkline
              }
          }
          formattedData.push(formattedItem);
        }
    });
    return formattedData;
}
export const getMarketData = async () => {
  try {
    const response = await axios.get(
      "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=20&page=1&sparkline=true&price_change_percentage=7d&locale=en"
    );
    const data = response.data;
    const formattedResponse = formatMarketData(data);
    return formattedResponse || [];
  } catch (error) {
    console.log("Error fetching market data:", error.message);
    return [];
  }
};
