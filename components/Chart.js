import { View, Text, StyleSheet, Image, Dimensions } from "react-native";

import React from "react";
import { LineChart, LineChartCursorCrosshair } from "react-native-wagmi-charts";
export const { width: SIZE } = Dimensions.get("window");
const Chart = ({
  currentPrice,
  logoUrl,
  name,
  symbol,
  priceChangePercentage7d,
  sparkline,
}) => {
  const priceChangeColor = priceChangePercentage7d > 0 ? "#34C759" : "#FF3B30";

  return (
    <LineChart.Provider data={sparkline}>
      <View style={styles.chartWrapper}>
        <View style={styles.titleWrapper}>
          <View style={styles.upperTitles}>
            <View style={styles.upperLeftTitle}>
              <Image source={{ url: logoUrl }} style={styles.image} />
              <Text style={styles.subTitle}>
                {name} ({symbol.toUpperCase()})
              </Text>
            </View>
            <Text style={styles.subTitle}>7d</Text>
          </View>

          <View style={styles.lowerTitles}>
          <LineChartCursorCrosshair >
<LineChart.PriceText  style={styles.boldTitle}
                format={(d) => {
                  'worklet';
                  return d.formatted ? `$${d.formatted} USD` : `$${currentPrice} USD`;
                }}
              />
</LineChartCursorCrosshair>
            <Text style={styles.boldTitle}>
      
              {/* ${currentPrice.toLocaleString("en-US", { currency: "USD" })} */}
            </Text>
            <Text style={[styles.title, { color: priceChangeColor }]}>
              {" "}
              {priceChangePercentage7d.toFixed(2)}%
            </Text>
          </View>
        </View>
        <View style={styles.chartLineWrapper}>
      
          <LineChart width={SIZE} height={SIZE / 2}>
            <LineChart.Path color="black" />
            <LineChart.CursorCrosshair>
              <LineChart.Tooltip
                textStyle={{
                  backgroundColor: "black",
                  borderRadius: 4,
                  color: "white",
                  fontSize: 18,
                  padding: 4,
                }}
              />
              
            </LineChart.CursorCrosshair>
          
          </LineChart>
        </View>
      </View>
    </LineChart.Provider>
  );
};

const styles = StyleSheet.create({
  chartWrapper: {
    marginVertical: 16,
  },
  titleWrapper: {
    marginHorizontal: 16,
  },
  upperTitles: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  upperLeftTitle: {
    flexDirection: "row",
    alignItems: "center",
  },
  image: {
    height: 24,
    width: 24,
    marginRight: 4,
  },
  subTitle: {
    fontSize: 14,
    color: "#A9ABB1",
  },
  lowerTitles: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  boldTitle: {
    fontSize: 24,
    fontWeight: "bold",
  },
  title: {
    fontSize: 18,
  },
  chartLineWrapper: {
    marginTop: 30,
  },
});
export default Chart;
