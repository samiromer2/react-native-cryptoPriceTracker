import { View, Text, StyleSheet, Image, Dimensions } from "react-native";

import React, { useEffect, useState } from "react";
import { LineChart, LineChartCursorCrosshair } from "react-native-wagmi-charts";
import { useSharedValue } from "react-native-reanimated";
export const { width: SIZE } = Dimensions.get("window");
const Chart = ({
  currentPrice,
  logoUrl,
  name,
  symbol,
  priceChangePercentage7d,
  sparkline,
}) => {
  const latestCurrentPrice = useSharedValue(currentPrice || 0);
  const [chartReady, setChartReady] = useState(false);

  const priceChangeColor = (priceChangePercentage7d || 0) > 0 ? "#10B981" : "#EF4444";
  const isPositive = (priceChangePercentage7d || 0) > 0;
  
  // Handle sparkline data - it might be array of numbers or array of objects
  const validSparkline = React.useMemo(() => {
    if (!sparkline || !Array.isArray(sparkline) || sparkline.length === 0) {
      return [];
    }
    // If first item is a number, it's raw data - return as is (will be handled by chart)
    // If first item is an object with timestamp/value, it's already formatted
    if (typeof sparkline[0] === 'number') {
      // Convert array of numbers to array of objects with timestamp and value
      const sevenDaysAgo = Math.floor(Date.now() / 1000) - (7 * 24 * 60 * 60);
      return sparkline.map((value, index) => ({
        timestamp: sevenDaysAgo + (index * 3600),
        value: value,
      }));
    }
    return sparkline;
  }, [sparkline]);

  useEffect(() => {
    if (currentPrice) {
      latestCurrentPrice.value = currentPrice;
    }
    // Small delay to ensure chart renders properly
    const timer = setTimeout(() => {
      setChartReady(true);
    }, 100);
    return () => clearTimeout(timer);
  }, [currentPrice, sparkline]);

  if (!validSparkline || validSparkline.length === 0) {
    return (
      <View style={styles.chartWrapper}>
        <View style={styles.headerSection}>
          <View style={styles.upperTitles}>
            <View style={styles.upperLeftTitle}>
              <View style={styles.imageContainer}>
                <Image source={{ uri: logoUrl }} style={styles.image} />
              </View>
              <View style={styles.coinInfo}>
                <Text style={styles.coinName} numberOfLines={1} ellipsizeMode="tail">
                  {name}
                </Text>
                <Text style={styles.coinSymbol}>{symbol?.toUpperCase()}</Text>
              </View>
            </View>
          </View>
        </View>
        <View style={styles.priceSection}>
          <View style={styles.priceContainer}>
            <Text style={styles.priceLabel}>Price</Text>
            <Text style={styles.boldTitle}>
              ${currentPrice?.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </Text>
          </View>
          <View style={styles.changeSection}>
            <Text style={styles.changeLabel}>Change</Text>
            <View style={[styles.changeBadge, { backgroundColor: isPositive ? "#D1FAE5" : "#FEE2E2" }]}>
              <Text style={[styles.changeText, { color: priceChangeColor }]}>
                {isPositive ? "↑" : "↓"} {Math.abs(priceChangePercentage7d || 0).toFixed(2)}%
              </Text>
            </View>
          </View>
        </View>
        <View style={styles.divider} />
        <View style={styles.chartPlaceholder}>
          <Text style={styles.placeholderText}>Chart data unavailable</Text>
        </View>
      </View>
    );
  }

  return (
    <LineChart.Provider data={validSparkline}>
      <View style={styles.chartWrapper}>
        {/* Header Section */}
        <View style={styles.headerSection}>
          <View style={styles.upperTitles}>
            <View style={styles.upperLeftTitle}>
              <View style={styles.imageContainer}>
                <Image source={{ uri: logoUrl }} style={styles.image} />
              </View>
              <View style={styles.coinInfo}>
                <Text style={styles.coinName} numberOfLines={1} ellipsizeMode="tail">
                  {name}
                </Text>
                <Text style={styles.coinSymbol}>{symbol.toUpperCase()}</Text>
              </View>
            </View>
            <View style={styles.timeBadge}>
              <Text style={styles.timeBadgeText}>7d</Text>
            </View>
          </View>
        </View>

        {/* Price Section */}
        <View style={styles.priceSection}>
          <View style={styles.priceContainer}>
            <Text style={styles.priceLabel}>Price</Text>
            <LineChartCursorCrosshair>
              <LineChart.PriceText
                style={styles.boldTitle}
                format={(d) => {
                  "worklet";
                  if (d && d.formatted) {
                    return `$${d.formatted}`;
                  }
                  return `$${currentPrice?.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                }}
              />
            </LineChartCursorCrosshair>
          </View>
          <View style={styles.changeSection}>
            <Text style={styles.changeLabel}>Change</Text>
            <View style={[styles.changeBadge, { backgroundColor: isPositive ? "#D1FAE5" : "#FEE2E2" }]}>
              <Text style={[styles.changeText, { color: priceChangeColor }]}>
                {isPositive ? "↑" : "↓"} {Math.abs(priceChangePercentage7d || 0).toFixed(2)}%
              </Text>
            </View>
          </View>
        </View>

        {/* Divider */}
        <View style={styles.divider} />

        {/* Chart Section */}
        <View style={styles.chartLineWrapper}>
          {chartReady && validSparkline.length > 0 ? (
            <View style={styles.chartContainer}>
              <LineChart width={SIZE - 64} height={220}>
                <LineChart.Path 
                  color={priceChangeColor}
                  strokeWidth={2.5}
                />
                <LineChart.CursorCrosshair>
                  <LineChart.Tooltip
                    textStyle={{
                      backgroundColor: "#1F2937",
                      borderRadius: 8,
                      color: "white",
                      fontSize: 14,
                      fontWeight: "600",
                      paddingHorizontal: 10,
                      paddingVertical: 6,
                    }}
                  />
                </LineChart.CursorCrosshair>
              </LineChart>
            </View>
          ) : (
            <View style={styles.chartPlaceholder}>
              <Text style={styles.placeholderText}>Loading chart...</Text>
            </View>
          )}
        </View>
      </View>
    </LineChart.Provider>
  );
};

const styles = StyleSheet.create({
  chartWrapper: {
    flex: 1,
    backgroundColor: "#FFFFFF",
  },
  headerSection: {
    paddingHorizontal: 24,
    paddingTop: 8,
    paddingBottom: 20,
  },
  upperTitles: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  upperLeftTitle: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
    marginRight: 12,
  },
  imageContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "#F9FAFB",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#F3F4F6",
  },
  image: {
    height: 40,
    width: 40,
  },
  coinInfo: {
    flex: 1,
  },
  coinName: {
    fontSize: 20,
    color: "#1F2937",
    fontWeight: "700",
    marginBottom: 4,
    letterSpacing: -0.3,
  },
  coinSymbol: {
    fontSize: 14,
    color: "#6B7280",
    fontWeight: "500",
  },
  timeBadge: {
    backgroundColor: "#F3F4F6",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  timeBadgeText: {
    fontSize: 13,
    color: "#374151",
    fontWeight: "600",
  },
  priceSection: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    paddingHorizontal: 24,
    paddingBottom: 20,
  },
  priceContainer: {
    flex: 1,
  },
  priceLabel: {
    fontSize: 12,
    color: "#6B7280",
    fontWeight: "500",
    marginBottom: 8,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  boldTitle: {
    fontSize: 32,
    fontWeight: "700",
    color: "#1F2937",
    letterSpacing: -0.8,
  },
  changeSection: {
    alignItems: "flex-end",
    marginLeft: 16,
  },
  changeLabel: {
    fontSize: 12,
    color: "#6B7280",
    fontWeight: "500",
    marginBottom: 8,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  changeBadge: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  changeText: {
    fontSize: 15,
    fontWeight: "700",
  },
  divider: {
    height: 1,
    backgroundColor: "#F3F4F6",
    marginHorizontal: 24,
    marginVertical: 4,
  },
  chartLineWrapper: {
    flex: 1,
    paddingTop: 20,
    paddingBottom: 8,
  },
  chartContainer: {
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 8,
  },
  chartPlaceholder: {
    height: 220,
    justifyContent: "center",
    alignItems: "center",
  },
  placeholderText: {
    fontSize: 14,
    color: "#9CA3AF",
    fontWeight: "500",
  },
  subTitle: {
    fontSize: 13,
    color: "#6B7280",
    fontWeight: "500",
  },
});
export default Chart;
